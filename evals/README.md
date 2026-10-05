# Artifact-based skill evaluations

Six small CPU tasks test actual repairs and evidence assessments. Inputs and prompts
live in `fixtures/`; private executable checks live in `checkers.py`. Prepared task
directories never contain the checkers, rubric, expected answers, or sibling attempts.

Start with [the evaluation guide](../docs/evaluation.md). The runner uses Python's
standard library and does not require an API key, network connection, or a particular
agent vendor. Your runtime reads `TASK_PROMPT_PATH`, works inside `TASK_WORKDIR`, and
writes `response.json` plus requested artifacts.

```bash
python evals/run.py prepare --case augmentation_stats --variant baseline --output /tmp/train-opt-baseline-01
python evals/run.py score --case augmentation_stats --workdir /tmp/train-opt-baseline-01
python evals/selftest.py
```

`selftest.py` uses synthetic test doubles to test this harness. It is **not** an agent
evaluation. `results/behavior-checks.json` preserves an older two-scenario manual
record, explicitly marked historical and not comparable to this suite. New actual
agent runs must state model/runtime, input version, conditions, and execution scope.
