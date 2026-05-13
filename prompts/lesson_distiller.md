You are distilling a *Failure Pattern* lesson from a verified before/after code transition in a Manim animation pipeline. The transition is *validated* — the after version was strictly better than the before version (either flipped a render error to success, or VLM score went up). Your job is to produce a structured Lesson that a future code generator can use as a hard constraint.

Output requirements: a single JSON object matching this schema (no markdown fence, no commentary):

```
{
  "trigger_pattern": "<plain-English condition under which this lesson applies, max 200 chars>",
  "root_cause": "<what was wrong in the before version, max 200 chars>",
  "fix_recipe": "<how to fix, written as an imperative instruction, max 200 chars>",
  "code_anti_example": "<minimal failing fragment from BEFORE code, max 400 chars>",
  "code_good_example": "<minimal repaired fragment from AFTER code, max 400 chars>",
  "vlm_diagnostic": "<copy or paraphrase of the original error / VLM remark, max 400 chars>"
}
```

Guidelines:
- `trigger_pattern` should generalize beyond the current scene — describe the visual / structural setup that triggered the bug, not the scene-specific text.
- `code_anti_example` and `code_good_example` should be short, runnable-ish fragments (2 to 5 lines) showing the specific construct that changed. Strip imports / boilerplate that's unrelated to the fix.
- If the transition was a render-error fix, the `vlm_diagnostic` should contain the key line from the traceback or LaTeX log.
- If the transition was a VLM-driven visual fix, the `vlm_diagnostic` should contain the revision_instruction or scoring feedback.

Do not invent reasons. If you cannot identify what changed, set `trigger_pattern` to "unknown transition" and copy the largest difference into the example fields.
