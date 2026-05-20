# P2M-Bench v2: Dataset Design for the EMB Snapshot Experiment

P2M-Bench v2 is a compact paper-section task stream for evaluating whether self-grown EMB reduces cross-task forgetting in Paper2Manim. The dataset separates a `memory_build` stream, used only to grow and snapshot the EMB, from a `fixed_probe` set, used for read-only evaluation of VLM Reflection Only and frozen EMB snapshots. The primary evidence is blind human scoring of first-attempt outputs, reported as Human Pass@1 and Human Quality Score, together with trace-derived Reflection Depth. Evaluation-only fields such as key claims, reference scene plans, task rubrics, and human scores are physically isolated from model inputs and EMB writes, ensuring that any observed improvement comes from self-grown memory rather than human-provided answers.

## 1. Core Experimental Claim

The dataset supports a narrow method claim:

> ManimAgent reduces cross-task forgetting by accumulating a self-grown dual-channel Episodic Memory Bank (EMB). As EMB snapshots become larger and more mature, the same VLM-reflection pipeline should produce better first-attempt paper-section animations: higher Human Pass@1, lower Reflection Depth, and higher Human Quality Score.

P2M-Bench v2 is therefore not framed as a broad visual education benchmark. It is a paper-section animation task stream for measuring first-attempt usability, paper alignment, visual robustness, and animation flow under controlled EMB snapshot conditions.

## 2. Main Experiment

### 2.1 Systems

The baseline is VLM Reflection Only:

```text
B = VLM Reflection Only
    --vlm --no-emb
```

The treatment systems use frozen EMB snapshots:

```text
C@K = VLM Reflection + frozen EMB snapshot with K records
      --vlm --emb --emb-readonly --emb-store-path <snapshot_K>
```

Main snapshot sizes:

```text
EMB@0
EMB@50
EMB@100
EMB@200
EMB@400
```

`K` is the number of consolidated EMB records, not necessarily the number of processed tasks.

### 2.2 Flow

Step A: Memory build

- Run ManimAgent on `split=memory_build` tasks.
- EMB writes are enabled.
- Save EMB snapshots when record count reaches `K in {0, 50, 100, 200, 400}`.

Step B: Fixed probe evaluation

- Run all systems on the same `split=fixed_probe` tasks.
- Snapshot systems use read-only memory with `--emb-readonly`.
- No probe output may be consolidated back into the snapshot.

Step C: Human evaluation

- Score first-attempt outputs from `B`, `C@0`, `C@100`, and `C@400`.
- Add `C@50` and `C@200` if annotation budget allows.
- Raters are blind to system condition.
- Conditions should contain the same fixed-probe task ids whenever possible.

### 2.3 Metrics

Primary metrics:

| metric | source | definition |
|---|---|---|
| `Human Pass@1` | human sidecar | Majority vote over `human_pass_at_1` on first-attempt outputs. |
| `Reflection Depth` | run trace | Mean number of revision/reflection rounds before pass or convergence. |
| `Human Quality Score` | human sidecar | Mean of five human dimensions. |

Auxiliary metrics:

| metric | source | definition |
|---|---|---|
| `VLM Quality Score` | run trace | Diagnostic only; not primary quality evidence. |
| `VLM-Human Agreement` | trace + sidecar | Pearson/Spearman for scores; kappa for pass/fail. |
| `Failure Flag Frequency` | human sidecar | Frequency of fatal or recurrent failure categories. |
| `Snapshot Size` | snapshot manifest | Number of consolidated EMB records. |

VLM score is auxiliary because the VLM is also used inside the system for revision and memory consolidation.

## 3. Task Table

The main table should be usable as a HuggingFace parquet schema and as a lightweight CSV for current runners.

### 3.1 Minimal CSV Schema

The runner-compatible CSV keeps a small required surface:

| column | type | required | notes |
|---|---|---:|---|
| `id` | str | recommended | Stable task id. If missing, derive from `arxiv_id + section`. |
| `arxiv_id` | str | yes | Bare arXiv id, e.g. `1706.03762`. |
| `section` | str | yes | Exact or near-exact section title passed to `paper2manim mvp2 --section`. |
| `domain` | enum | yes | `cs / math / physics / quantum / econ` initially. |
| `split` | enum | yes | v2 split name or supported legacy alias. |
| `expected_scene_count_min` | int | optional | Sanity floor only. |

Recommended extra CSV columns for v2 experiment planning:

```text
stream_idx
probe_idx
scene_role
category
difficulty
human_eval_candidate
```

### 3.2 Canonical Splits

| split | purpose |
|---|---|
| `memory_build` | Ordered stream used to grow EMB and save snapshots. |
| `fixed_probe` | Held-out tasks used for read-only evaluation of baseline and snapshots. |
| `test_holdout` | Reserved final set; do not tune on it. |
| `cross_train` | Optional appendix: grow EMB on one domain. |
| `cross_test` | Optional appendix: frozen cross-domain probe. |

Legacy aliases are allowed for current lightweight scripts:

| legacy split | canonical split |
|---|---|
| `bootstrap` | `memory_build` |
| `eval` | `fixed_probe` |
| `cross_train` | `cross_train` |
| `cross_test` | `cross_test` |

Documentation, figures, and new dataset releases should use `memory_build` and `fixed_probe`.

### 3.3 Full Parquet Fields

| field | type | required | visibility | purpose |
|---|---|---:|---|---|
| `id` | str | yes | model/eval | Stable task id. |
| `paper_id` | str | yes | model/eval | Same as arXiv id or local PDF SHA. |
| `arxiv_id` | str | yes for arXiv | model/eval | CLI-compatible id. |
| `paper_title` | str | yes | model/eval | Display and annotation context. |
| `paper_publish_date` | date | yes | eval metadata | Contamination analysis. |
| `paper_license` | str | yes | release metadata | Dataset card/license. |
| `paper_full_text` | str | yes in parquet | model input | Cleaned paper text. |
| `target_unit` | struct | yes | model/eval | Local paper unit to animate. |
| `section` | str | yes in CSV | model/eval | CLI section selector. |
| `source_type` | enum | yes | model/eval | Main experiment: `section`. |
| `category` | enum | yes | eval stratification | `Concept / Equation / Algorithm / Figure / Architecture / Experiment`. |
| `scene_role` | enum | yes | model/eval | `BACKGROUND / METHOD / EXPERIMENT / CONCLUSION`. |
| `domain` | enum | yes | model/eval | Used by CLI `--dataset-domain` and EMB context. |
| `difficulty` | enum | yes | eval stratification | `easy / medium / hard`. |
| `track` | enum | yes | eval filter | Main experiment uses `local`. |
| `split` | enum | yes | runner/eval | `memory_build / fixed_probe / test_holdout / cross_train / cross_test`. |
| `stream_idx` | int/null | required for `memory_build` | runner/eval | Frozen order for memory-building stream. |
| `probe_idx` | int/null | required for `fixed_probe` | runner/eval | Frozen order for probe evaluation. |
| `human_eval_candidate` | bool | yes | eval sampling | Eligible for human scoring. |
| `main_topics` | list[str] | yes | eval-only | Rater aid; never model input. |
| `key_claims` | list[str] | yes | eval-only | Rater aid; never model input. |
| `reference_scene_plan` | list[object] | yes | eval-only | Rater aid; not a mandatory script. |
| `human_rubric` | object | yes | eval-only | Task-specific scoring reminders. |
| `contamination_group` | enum/null | optional | eval | `pre_cutoff / post_cutoff / unknown`. |
| `notes` | str | optional | eval | Annotation notes. |

`category` describes the intended animation content. `source_type` describes the paper artifact. In the main experiment, `source_type=section`, `target_unit.type=section`, and `track=local`.

### 3.4 `target_unit`

```json
{
  "title": "Section 3.2.1 Scaled Dot-Product Attention",
  "text": "<target excerpt, <= 4000 chars>",
  "type": "section",
  "anchor": {
    "section": "3.2.1",
    "equation_label": null,
    "figure_label": null,
    "page": null
  },
  "required_prior_context": "",
  "prior_objects": []
}
```

CSV maps `section` to `target_unit.title` or `target_unit.anchor.section`, depending on the loader. For the main experiment, `target_unit.text <= 4000 chars`. `full_paper` remains outside the main experiment and may be documented only as an appendix extension.

## 4. Field Isolation

Strict field isolation is part of the experimental design.

### 4.1 Model-Visible Fields

Only these fields may enter the model prompt, Storyboarder, Coder, VLM reflection context, or EMB query:

```text
paper_full_text
target_unit.title
target_unit.text
target_unit.type
target_unit.anchor
target_unit.required_prior_context
target_unit.prior_objects
scene_role
domain
source_type
paper_title
```

### 4.2 Evaluation-Only Fields

These fields are evaluation-only and must never enter model input or EMB:

```text
main_topics
key_claims
reference_scene_plan
human_rubric
human_scores
human_pass
fatal_flags
human_quality_score
```

This rule is critical: the paper's claim is self-grown memory. EMB must be produced by the system's own task stream, not by human-provided answer fields.

### 4.3 EMB Context Mapping

Only safe fields map into EMB context:

| dataset field | EMB context |
|---|---|
| `target_unit.text` | `task_text` |
| `scene_role` | `scene_role` |
| `domain` | `domain_tags` |
| `paper_id` / `arxiv_id` | `source_paper` |
| `target_unit.anchor.section` | `source_section` |

Never write `key_claims`, `reference_scene_plan`, `human_rubric`, fatal flags, or human scores into EMB.

## 5. Human Scoring Sidecar

Human scoring is output-level, not task-level. Store it in a separate JSONL or parquet sidecar with one row per `(output_id, rater_id_hash)`.

### 5.1 Scored Outputs

The primary human evaluation scores first-attempt videos. Final converged outputs are optional appendix material, because the main claim is that EMB improves initial generation before revision.

### 5.2 Scoring Dimensions

Use five intro-aligned dimensions:

1. `paper_alignment`: Does the animation faithfully represent the target paper unit?
2. `key_claim_coverage`: Does it cover the key claims and central ideas?
3. `visual_robustness`: Is it readable and free of occlusion, overlap, cropping, unreadable text, or broken layout?
4. `animation_flow`: Does it unfold in a reasonable order?
5. `first_attempt_usability`: Is this first attempt usable without major repair?

Binary label:

```text
human_pass_at_1: yes/no
```

Aggregate:

```text
Human Quality Score =
mean(paper_alignment,
     key_claim_coverage,
     visual_robustness,
     animation_flow,
     first_attempt_usability)
```

### 5.3 Fatal Flags

```text
empty_or_unplayable
unrelated_to_target
major_formula_or_symbol_error
unsupported_hallucination
missing_central_idea
severe_occlusion_or_overlap
cropped_or_offscreen
unreadable_text
text_only_or_weak_visualization
incoherent_animation_order
other
```

### 5.4 Sidecar Example

Do not expose `condition_hidden` to raters. It is for aggregation only.

```json
{
  "task_id": "1706.03762_method_3_2_1",
  "output_id": "run_xxx_scene_00_attempt_0",
  "run_id": "run_xxx",
  "condition_blind_id": "A17",
  "condition_hidden": {
    "system": "C",
    "snapshot_records": 100
  },
  "attempt": "attempt_0",
  "video_path": "runs/.../scene_00_attempt_0.mp4",
  "rater_id_hash": "rater_03",
  "scores": {
    "paper_alignment": 4,
    "key_claim_coverage": 4,
    "visual_robustness": 3,
    "animation_flow": 4,
    "first_attempt_usability": 3
  },
  "human_pass_at_1": false,
  "fatal_flags": {
    "empty_or_unplayable": false,
    "unrelated_to_target": false,
    "major_formula_or_symbol_error": false,
    "unsupported_hallucination": false,
    "missing_central_idea": false,
    "severe_occlusion_or_overlap": false,
    "cropped_or_offscreen": false,
    "unreadable_text": false,
    "text_only_or_weak_visualization": false,
    "incoherent_animation_order": false,
    "other": false
  },
  "comment": "",
  "time_spent_sec": 90
}
```

## 6. EMB Snapshot Manifest

Snapshot metadata is separate from the task table:

```json
{
  "experiment_id": "exp_main_2026_xxx",
  "seed": 1,
  "snapshot_id": "C_seed1_records100",
  "config": "C",
  "emb_store_path": "runs/exp_main/snapshots/seed_1/records_100",
  "record_count_total": 100,
  "record_count_success": 45,
  "record_count_failure": 55,
  "source_task_count": 37,
  "last_memory_build_task_id": "xxxx",
  "created_at": "2026-05-21T00:00:00Z"
}
```

Fixed-probe evaluation must use snapshots in read-only mode:

```text
--emb-readonly
```

No probe output may be consolidated back into the snapshot.

## 7. Release and Contamination Notes

The task table keeps `paper_publish_date`, `paper_license`, and `contamination_group` so results can be reported by pre-cutoff, post-cutoff, and unknown buckets. arXiv papers may appear in pretraining corpora; date stratification mitigates contamination concerns but does not prove a model has not seen a paper.

The dataset release should include:

| artifact | purpose |
|---|---|
| Task table | Model-visible task stream plus eval metadata with isolation rules. |
| Model input view | Physically removes eval-only fields. |
| Evaluation sidecar | Key claims, reference scene plans, and human rubrics. |
| Human scoring sidecar | Blind output-level first-attempt scores. |
| Snapshot manifest | Frozen EMB snapshot metadata. |
| Dataset card | License, source, contamination, and responsible release notes. |

## 8. Appendix Extensions

The following are optional extensions, not main-body dataset requirements:

- `full_paper` or global narrative tasks.
- Cross-domain transfer beyond `cross_train` and `cross_test`.
- Curriculum variants.
- Isomorphic task pairs.
- Human ceiling studies.
- Storyboard alignment as a separate metric.

They should not replace the main experiment: VLM Reflection Only vs frozen EMB snapshots on the same fixed probe.
