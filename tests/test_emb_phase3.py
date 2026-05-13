"""Phase 3 tests: RAG retrieval + Coder prompt injection."""

from __future__ import annotations

from paper2manim.agents.coder import _build_user_prompt, coder_node
from paper2manim.emb import (
    Context,
    FailureBody,
    MemoryRecord,
    Provenance,
    SuccessBody,
)
from paper2manim.emb.manager import build_in_memory_emb
from paper2manim.emb.retrieval import (
    RetrievalBundle,
    render_known_pitfalls_block,
    render_reference_examples_block,
    retrieve_for_scene,
    summarize_bundle,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _success_rec(text: str, scene_id: str = "S") -> MemoryRecord:
    return MemoryRecord(
        polarity="success",
        context=Context(task_text=text, source_paper="arxiv:test", source_section="Background"),
        body=SuccessBody(rationale="r", code_full="from manim import *\nclass S(Scene): pass"),
        provenance=Provenance(scene_id=scene_id, validated=True, vlm_score=4.5),
    )


def _failure_rec(text: str, scene_id: str = "S") -> MemoryRecord:
    return MemoryRecord(
        polarity="failure",
        context=Context(task_text=text, source_paper="arxiv:test", source_section="Background"),
        body=FailureBody(
            trigger_pattern="trig",
            root_cause="rc",
            fix_recipe="fr",
            code_anti_example="bad()",
            code_good_example="good()",
        ),
        provenance=Provenance(
            scene_id=scene_id,
            validated=True,
            before_score=2.0,
            after_score=3.5,
            extraction_source="visual_reflection",
        ),
    )


# --------------------------------------------------------------------------- #
# retrieve_for_scene
# --------------------------------------------------------------------------- #


class TestRetrieveForScene:
    def test_returns_bundle_with_both_polarities(self):
        emb = build_in_memory_emb()
        emb.put(_success_rec("triangle theorem", "Intro"))
        emb.put(_failure_rec("axes overlap", "Axes"))
        bundle = retrieve_for_scene(emb, "triangle scene", k_success=2, k_failure=2)
        assert isinstance(bundle, RetrievalBundle)
        assert len(bundle.success) == 1
        assert len(bundle.failure) == 1

    def test_empty_emb_returns_empty_bundle(self):
        emb = build_in_memory_emb()
        bundle = retrieve_for_scene(emb, "anything", k_success=3, k_failure=3)
        assert bundle.is_empty()

    def test_k_zero_returns_empty(self):
        emb = build_in_memory_emb()
        emb.put(_success_rec("x"))
        emb.put(_failure_rec("y"))
        bundle = retrieve_for_scene(emb, "anything", k_success=0, k_failure=0)
        assert bundle.success == []
        assert bundle.failure == []

    def test_query_failure_degrades_gracefully(self):
        """If the embedder raises, retrieval returns an empty bundle (no exception)."""

        class BrokenEmbedder:
            dim = 8

            def encode_one(self, text: str):
                raise RuntimeError("boom")

            def encode(self, texts):
                raise RuntimeError("boom")

        from paper2manim.emb.index import InMemoryVectorIndex
        from paper2manim.emb.manager import EpisodicMemoryBank
        from paper2manim.emb.store import InMemoryMemoryStore

        emb = EpisodicMemoryBank(
            store=InMemoryMemoryStore(),
            embedder=BrokenEmbedder(),
            success_index=InMemoryVectorIndex(8),
            failure_index=InMemoryVectorIndex(8),
        )
        bundle = retrieve_for_scene(emb, "x", k_success=1, k_failure=1)
        assert bundle.is_empty()

    def test_to_state_dict_strips_embedding(self):
        emb = build_in_memory_emb()
        emb.put(_success_rec("x"))
        bundle = retrieve_for_scene(emb, "x", k_success=1, k_failure=0)
        wire = bundle.to_state_dict()
        assert "task_embedding" not in wire["success"][0]["context"]


# --------------------------------------------------------------------------- #
# Block rendering
# --------------------------------------------------------------------------- #


class TestRenderBlocks:
    def test_empty_records_yield_empty_string(self):
        assert render_reference_examples_block([]) == ""
        assert render_known_pitfalls_block([]) == ""

    def test_reference_examples_includes_rationale_and_code(self):
        emb = build_in_memory_emb()
        emb.put(_success_rec("triangle scene", "Intro"))
        bundle = retrieve_for_scene(emb, "triangle", k_success=1, k_failure=0)
        block = render_reference_examples_block(bundle.success)
        assert "Reference Examples" in block
        assert "Rationale:" in block
        assert "from manim import *" in block
        assert "arxiv:test" in block
        assert "Intro" in block

    def test_known_pitfalls_includes_anti_and_good(self):
        emb = build_in_memory_emb()
        emb.put(_failure_rec("axes overlap", "Axes"))
        bundle = retrieve_for_scene(emb, "axes", k_success=0, k_failure=1)
        block = render_known_pitfalls_block(bundle.failure)
        assert "Known Pitfalls" in block
        assert "Anti-example" in block
        assert "Good example" in block
        assert "bad()" in block
        assert "good()" in block

    def test_render_accepts_wire_dicts(self):
        emb = build_in_memory_emb()
        emb.put(_success_rec("x"))
        bundle = retrieve_for_scene(emb, "x", k_success=1, k_failure=0)
        wire = bundle.to_state_dict()
        # Render directly from wire-format dicts (the path used by the graph).
        block = render_reference_examples_block(wire["success"])
        assert "Reference Examples" in block

    def test_render_truncates_long_code(self):
        long_code = "x = 1\n" * 1000  # well over 1200 chars
        rec = MemoryRecord(
            polarity="success",
            context=Context(task_text="t"),
            body=SuccessBody(rationale="r", code_full=long_code),
            provenance=Provenance(scene_id="S"),
        )
        emb = build_in_memory_emb()
        emb.put(rec)
        bundle = retrieve_for_scene(emb, "t", k_success=1, k_failure=0)
        block = render_reference_examples_block(bundle.success)
        assert "..." in block
        # Block should be much shorter than the original code
        assert len(block) < len(long_code)

    def test_summarize_bundle_is_compact(self):
        emb = build_in_memory_emb()
        emb.put(_success_rec("a"))
        emb.put(_failure_rec("b"))
        bundle = retrieve_for_scene(emb, "x", k_success=1, k_failure=1)
        summary = summarize_bundle(bundle)
        assert set(summary.keys()) == {"success", "failure"}
        assert all("similarity" in item and "id" in item for item in summary["success"])


# --------------------------------------------------------------------------- #
# Coder integration
# --------------------------------------------------------------------------- #


class TestCoderPromptInjection:
    """Verify _build_user_prompt inlines blocks when state carries records."""

    def _base_state(self) -> dict:
        return {
            "storyboard": {
                "title": "T",
                "scenes": [
                    {"name": "S1", "description": "a scene description",
                     "duration_hint": 5.0}
                ],
            },
            "current_scene_idx": 0,
            "iter_count": 0,
        }

    def test_prompt_without_records_omits_blocks(self):
        prompt = _build_user_prompt(self._base_state())
        assert "Reference Examples" not in prompt
        assert "Known Pitfalls" not in prompt

    def test_prompt_with_success_records_includes_ref_block(self):
        emb = build_in_memory_emb()
        emb.put(_success_rec("triangle theorem", "Intro"))
        bundle = retrieve_for_scene(emb, "triangle", k_success=1, k_failure=0)
        state = self._base_state()
        state["retrieved_success"] = bundle.to_state_dict()["success"]
        prompt = _build_user_prompt(state)
        assert "Reference Examples" in prompt
        assert "Known Pitfalls" not in prompt
        assert "from manim import *" in prompt

    def test_prompt_with_failure_records_includes_pitfalls_block(self):
        emb = build_in_memory_emb()
        emb.put(_failure_rec("axes overlap", "Axes"))
        bundle = retrieve_for_scene(emb, "axes", k_success=0, k_failure=1)
        state = self._base_state()
        state["retrieved_failure"] = bundle.to_state_dict()["failure"]
        prompt = _build_user_prompt(state)
        assert "Known Pitfalls" in prompt
        assert "Reference Examples" not in prompt
        assert "bad()" in prompt

    def test_prompt_with_both_polarities(self):
        emb = build_in_memory_emb()
        emb.put(_success_rec("a", "Sa"))
        emb.put(_failure_rec("b", "Sb"))
        bundle = retrieve_for_scene(emb, "x", k_success=1, k_failure=1)
        wire = bundle.to_state_dict()
        state = self._base_state()
        state["retrieved_success"] = wire["success"]
        state["retrieved_failure"] = wire["failure"]
        prompt = _build_user_prompt(state)
        assert "Reference Examples" in prompt
        assert "Known Pitfalls" in prompt

    def test_empty_lists_do_not_inject_blocks(self):
        state = self._base_state()
        state["retrieved_success"] = []
        state["retrieved_failure"] = []
        prompt = _build_user_prompt(state)
        assert "Reference Examples" not in prompt
        assert "Known Pitfalls" not in prompt


class TestCoderNodeUsesRetrieval:
    """End-to-end: coder_node calls the mock LLM with the augmented prompt."""

    def test_coder_node_passes_retrieval_to_llm(self, mock_llm, fake_storyboard):
        mock_llm.invoke.return_value.content = "```python\nfrom manim import *\nclass PythagorasIntro(Scene):\n    def construct(self): pass\n```"
        state = {
            "run_id": "rid_coder_inject",
            "storyboard": fake_storyboard,
            "current_scene_idx": 0,
            "iter_count": 0,
            "retrieved_success": [
                {
                    "id": "x",
                    "polarity": "success",
                    "similarity": 0.9,
                    "context": {"source_paper": "arxiv:foo", "source_section": "Method"},
                    "body": {"rationale": "test rationale", "code_full": "MARKER_TEST_CODE"},
                    "provenance": {"scene_id": "PriorScene"},
                }
            ],
            "retrieved_failure": [],
        }
        coder_node(state)
        # The invoked LLM should have received a user prompt mentioning the
        # retrieved example.
        call_args, _ = mock_llm.invoke.call_args
        messages = call_args[0]  # first positional: list of (role, content)
        # The user prompt is the last message; concat all for safety
        joined = "\n".join(content for _, content in messages)
        assert "Reference Examples" in joined
        assert "MARKER_TEST_CODE" in joined
        assert "test rationale" in joined
