"""Durable record store. Default impl: SQLite (WAL) with a single table.

The store owns the persistent ground truth for every :class:`MemoryRecord`. The
vector index (separate file) caches just the embeddings for ANN; if it gets
corrupted we can rebuild it by re-encoding every record's ``context.task_text``
fetched from this store.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from paper2manim.emb.exceptions import RecordNotFoundError, StoreError
from paper2manim.emb.schema import (
    Context,
    FailureBody,
    MemoryRecord,
    Polarity,
    Provenance,
    SuccessBody,
)

log = logging.getLogger(__name__)


@runtime_checkable
class MemoryStore(Protocol):
    """Durable, polarity-aware record store."""

    def put(self, record: MemoryRecord) -> None: ...

    def get(self, record_id: str) -> MemoryRecord: ...

    def delete(self, record_id: str) -> None: ...

    def all(self, *, polarity: Polarity | None = None) -> list[MemoryRecord]: ...

    def count(self, *, polarity: Polarity | None = None) -> int: ...

    def bump_hit(self, record_id: str, *, now: float | None = None) -> None: ...

    def find_id_by_provenance(
        self,
        run_id: str,
        scene_id: str,
        polarity: Polarity,
        extraction_source: str,
        transition_ordinal: int = 0,
    ) -> str | None: ...


# ---- SQLite default ----


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_records (
    id                 TEXT PRIMARY KEY,
    polarity           TEXT NOT NULL CHECK (polarity IN ('success','failure')),
    run_id             TEXT NOT NULL DEFAULT '',
    scene_id           TEXT NOT NULL DEFAULT '',
    extraction_source  TEXT NOT NULL DEFAULT '',
    transition_ordinal INTEGER NOT NULL DEFAULT 0,
    -- Split-level dataset domain (cs / math / physics / quantum / econ).
    -- Pulled out of context_json into its own column so ``WHERE domain = ?``
    -- queries don't have to parse every record. Default '' = "unknown".
    domain             TEXT NOT NULL DEFAULT '',
    context_json       TEXT NOT NULL,
    body_json          TEXT NOT NULL,
    provenance_json    TEXT NOT NULL,
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_polarity ON memory_records(polarity);
CREATE INDEX IF NOT EXISTS idx_created  ON memory_records(created_at);
-- idx_domain is created by _MIGRATIONS_SQL so legacy DBs without the
-- ``domain`` column don't fail this CREATE INDEX before the ALTER TABLE runs.
-- Partial unique index: rows with empty provenance (legacy / test scaffolding)
-- are exempt; rows with real (run_id, scene_id) get deduped per
-- (polarity, extraction_source, transition_ordinal). ``transition_ordinal``
-- preserves within-scene ordering for failure transitions so v0→v1 and v1→v2
-- of the same scene don't collide.
CREATE UNIQUE INDEX IF NOT EXISTS uq_provenance
    ON memory_records(run_id, scene_id, polarity, extraction_source, transition_ordinal)
    WHERE run_id != '' AND scene_id != '';
"""


_MIGRATIONS_SQL = [
    # Idempotent: ALTER TABLE ADD COLUMN raises if the column already exists, so
    # we guard each one. Older databases created before the provenance columns
    # existed need these to be backfilled before the unique index is added.
    "ALTER TABLE memory_records ADD COLUMN run_id TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE memory_records ADD COLUMN scene_id TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE memory_records ADD COLUMN extraction_source TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE memory_records ADD COLUMN transition_ordinal INTEGER NOT NULL DEFAULT 0",
    # B6: domain column for RQ3 cross-domain freeze. Default '' keeps pre-B6
    # records readable; new writes populate from Context.domain.
    "ALTER TABLE memory_records ADD COLUMN domain TEXT NOT NULL DEFAULT ''",
    "CREATE INDEX IF NOT EXISTS idx_domain ON memory_records(domain)",
    # Replace the old 4-tuple uq_provenance with the new 5-tuple version on
    # legacy databases. Safe to run on fresh DBs too (drops then re-creates
    # the identical index). Existing 4-tuple-deduped data extends cleanly
    # under transition_ordinal=0.
    "DROP INDEX IF EXISTS uq_provenance",
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_provenance "
        "ON memory_records(run_id, scene_id, polarity, extraction_source, transition_ordinal) "
        "WHERE run_id != '' AND scene_id != ''"
    ),
]


class SQLiteMemoryStore:
    """Single-file SQLite-backed store with WAL mode.

    Thread-safe via a single shared connection guarded by a lock. We expect
    write throughput to be modest (1–2 inserts per scene), so a single writer
    is fine; readers benefit from WAL snapshots.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path), detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA_SQL)
            # Apply add-column migrations for databases predating the
            # provenance columns. Silently swallow "duplicate column" errors.
            for stmt in _MIGRATIONS_SQL:
                try:
                    self._conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass
            self._conn.commit()

    # ---- helpers ----

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        try:
            polarity: Polarity = row["polarity"]
            ctx = Context.model_validate_json(row["context_json"])
            body_blob = row["body_json"]
            body: SuccessBody | FailureBody
            if polarity == "success":
                body = SuccessBody.model_validate_json(body_blob)
            else:
                body = FailureBody.model_validate_json(body_blob)
            prov = Provenance.model_validate_json(row["provenance_json"])
        except Exception as exc:  # noqa: BLE001
            raise StoreError(
                f"Corrupted record id={row['id']!r}: {type(exc).__name__}: {exc}"
            ) from exc
        return MemoryRecord(id=row["id"], polarity=polarity, context=ctx, body=body, provenance=prov)

    # ---- API ----

    def find_id_by_provenance(
        self,
        run_id: str,
        scene_id: str,
        polarity: Polarity,
        extraction_source: str,
        transition_ordinal: int = 0,
    ) -> str | None:
        """Return the existing record id matching the provenance tuple, if any.

        Used by the manager to merge re-consolidations of the same
        (run_id, scene_id, polarity, extraction_source, transition_ordinal)
        instead of accumulating duplicates. Empty run_id / scene_id are not
        deduped (see partial index). The ``transition_ordinal`` argument keeps
        within-scene transitions (v0→v1 vs v1→v2) distinct.
        """
        if not run_id or not scene_id:
            return None
        with self._lock:
            cur = self._conn.execute(
                "SELECT id FROM memory_records "
                "WHERE run_id=? AND scene_id=? AND polarity=? "
                "AND extraction_source=? AND transition_ordinal=?",
                (run_id, scene_id, polarity, extraction_source, transition_ordinal),
            )
            row = cur.fetchone()
        return row["id"] if row else None

    def put(self, record: MemoryRecord) -> None:
        now = time.time()
        prov = record.provenance
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memory_records "
                    "(id, polarity, run_id, scene_id, extraction_source, "
                    " transition_ordinal, domain, context_json, body_json, "
                    " provenance_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "COALESCE((SELECT created_at FROM memory_records WHERE id=?), ?), ?)",
                    (
                        record.id,
                        record.polarity,
                        prov.run_id,
                        prov.scene_id,
                        prov.extraction_source,
                        prov.transition_ordinal,
                        record.context.domain,
                        record.context.model_dump_json(),
                        record.body.model_dump_json(),
                        prov.model_dump_json(),
                        record.id,
                        now,
                        now,
                    ),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                raise StoreError(f"SQLite write failed: {exc}") from exc

    def get(self, record_id: str) -> MemoryRecord:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM memory_records WHERE id = ?", (record_id,)
            )
            row = cur.fetchone()
        if row is None:
            raise RecordNotFoundError(record_id)
        return self._row_to_record(row)

    def delete(self, record_id: str) -> None:
        with self._lock:
            try:
                self._conn.execute("DELETE FROM memory_records WHERE id = ?", (record_id,))
                self._conn.commit()
            except sqlite3.Error as exc:
                raise StoreError(f"SQLite delete failed: {exc}") from exc

    def all(self, *, polarity: Polarity | None = None) -> list[MemoryRecord]:
        with self._lock:
            if polarity is None:
                cur = self._conn.execute(
                    "SELECT * FROM memory_records ORDER BY created_at ASC"
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM memory_records WHERE polarity = ? ORDER BY created_at ASC",
                    (polarity,),
                )
            rows: Iterable[sqlite3.Row] = cur.fetchall()
        return [self._row_to_record(r) for r in rows]

    def count(self, *, polarity: Polarity | None = None) -> int:
        with self._lock:
            if polarity is None:
                cur = self._conn.execute("SELECT COUNT(*) AS n FROM memory_records")
            else:
                cur = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM memory_records WHERE polarity = ?",
                    (polarity,),
                )
            return int(cur.fetchone()["n"])

    def bump_hit(self, record_id: str, *, now: float | None = None) -> None:
        ts = now if now is not None else time.time()
        try:
            rec = self.get(record_id)
        except RecordNotFoundError:
            log.warning("[store] bump_hit on missing id=%s", record_id)
            return
        rec.provenance.hit_count += 1
        rec.provenance.last_used = ts
        self.put(rec)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


# ---- In-memory fallback (tests / scratch) ----


class InMemoryMemoryStore:
    """Dict-backed store. Useful for tests; not durable across process restarts."""

    def __init__(self) -> None:
        self._data: dict[str, MemoryRecord] = {}

    def put(self, record: MemoryRecord) -> None:
        self._data[record.id] = record.model_copy(deep=True)

    def get(self, record_id: str) -> MemoryRecord:
        if record_id not in self._data:
            raise RecordNotFoundError(record_id)
        return self._data[record_id].model_copy(deep=True)

    def delete(self, record_id: str) -> None:
        self._data.pop(record_id, None)

    def all(self, *, polarity: Polarity | None = None) -> list[MemoryRecord]:
        items = list(self._data.values())
        if polarity is not None:
            items = [r for r in items if r.polarity == polarity]
        items.sort(key=lambda r: r.provenance.first_seen)
        return [r.model_copy(deep=True) for r in items]

    def count(self, *, polarity: Polarity | None = None) -> int:
        if polarity is None:
            return len(self._data)
        return sum(1 for r in self._data.values() if r.polarity == polarity)

    def bump_hit(self, record_id: str, *, now: float | None = None) -> None:
        if record_id not in self._data:
            return
        rec = self._data[record_id]
        rec.provenance.hit_count += 1
        rec.provenance.last_used = now if now is not None else time.time()

    def find_id_by_provenance(
        self,
        run_id: str,
        scene_id: str,
        polarity: Polarity,
        extraction_source: str,
        transition_ordinal: int = 0,
    ) -> str | None:
        if not run_id or not scene_id:
            return None
        for rid, rec in self._data.items():
            p = rec.provenance
            if (
                p.run_id == run_id
                and p.scene_id == scene_id
                and rec.polarity == polarity
                and p.extraction_source == extraction_source
                and p.transition_ordinal == transition_ordinal
            ):
                return rid
        return None
