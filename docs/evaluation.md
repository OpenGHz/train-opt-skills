# Evaluating the skill

The evaluation suite measures whether an agent repairs small known defects or
draws the supported conclusion from supplied evidence. Its CPU checks do not prove
production training correctness, GPU speedup, deployment quality, or exact resume.
Unit tests of the tools and agent evaluations are separate evidence categories.

## Cases and required artifacts

| Case | Task | What the private checker exercises |
|---|---|---|
| `async_timing` | Repair `pipeline.py` measurement helper | Completed asynchronous window, warmup exclusion, boundary-only synchronization, invalid windows |
| `duplicate_shards` | Repair rank × worker assignment | Unseen dataset sizes/topologies, exact coverage, empty shards, invalid topology |
| `augmentation_stats` | Repair normalization | Per-image population statistics, independence from batch members, constant/empty inputs, no mutation |
| `weight_reorder` | Repair QKV conversion | Unseen head counts, same-shape layout semantics, projection result, malformed inputs |
| `protocol_change` | Write `assessment.json` | Observed rate arithmetic, changed camera/resolution contract, unmeasured quality and time-to-target |
| `optional_integration` | Produce fallback audit and integration record | Missing-specialist status, data-contract failure, executable audit on unseen IDs |

All cases require `response.json` matching the schema provided equally to both
conditions. Missing or invalid responses fail, even if the runtime exits zero.
Code-repair tasks test the repaired callable, rather than explanatory keywords.
Assessment tasks test structured numeric and logical conclusions. These checks
cannot establish that every sentence of free-text reasoning is correct; retain
agent traces for qualitative review.

## Prepare, run manually, and score

Run commands from the project root. Choose a new destination **outside this source
project**; the runner refuses existing paths and never overwrites an attempt.

```bash
python evals/run.py prepare --case augmentation_stats --variant baseline --output /tmp/train-opt-a01
python evals/run.py prepare --case augmentation_stats --variant treatment --output /tmp/train-opt-b01
```

Start a fresh agent session for each directory and give it only that directory's
`TASK.md`. Ask it to work there and finish the requested files. Baseline contains
only fixture inputs, a prompt, a response schema, and preparation metadata.
Treatment additionally contains `_skill/optimize-training/` and a prompt instruction
to read it. Prompts, source fixtures, and skill files are hashed in
`.eval/manifest.json`. Do not give either agent the checkers or another attempt.

```bash
python evals/run.py score --case augmentation_stats --workdir /tmp/train-opt-a01
python evals/run.py score --case augmentation_stats --workdir /tmp/train-opt-b01
```

The command prints JSON, saves `.eval/score.json`, and returns 0 for pass or 1 for
failure. Fixture inputs are checked against the authoritative harness files, not
only participant-writable metadata. Metadata-directory and report symlinks are
rejected. `pipeline.py` is the intended
mutable repair artifact. Keep a manual run's runtime/model identity, agent transcript,
tool traces, resource budget, and timestamps beside the results. Preparing a task
is `not_run`; only execution plus scoring counts as an evaluated attempt. A score
file itself does not authenticate a model identity.

## Bring your own runtime

`run` executes a JSON array of arguments with `shell=False`; it does not substitute
shell variables or interpret shell syntax. The runtime may be an installed agent
CLI wrapper or your own adapter. It must read these environment variables:

| Variable | Value |
|---|---|
| `TASK_PROMPT_PATH` | Absolute path to `TASK.md` |
| `TASK_WORKDIR` | Absolute task directory; also the process working directory |
| `EXEC_VARIANT` | `baseline` or `treatment` |
| `TASK_SKILL_PATH` | Absolute core-skill directory, **treatment only**; unset for baseline |

For example, after implementing your runtime adapter:

```bash
python evals/run.py run --runtime-json '["python3", "/absolute/path/to/your_runtime_adapter.py"]' --variant both --output /tmp/train-opt-ab01 --timeout 300
```

Use repeated `--case CASE` to select a subset. The adapter must perform the task,
save artifacts, and return; printing a suggested answer is insufficient. Do not put
secrets in arguments because runtime arguments are recorded. Credentials and network
access are not built into this project. The runner inherits the caller's environment
so an independently configured runtime can operate.

Every attempt preserves `.eval/stdout.txt`, `.eval/stderr.txt`, `.eval/runtime.json`,
and its score; the run root contains `summary.json`. Runtime metadata records argv,
timeout, timestamps, return code, and completion/timeout/launch-failure status.
POSIX timeouts kill the runtime process group; other platforms kill the direct
process only. Nonzero runtime exits and timeouts cannot count as successful attempts.
Checker execution has a separate timeout (15 seconds by default).
Both timeout settings must be finite and positive. Invalid settings are rejected
before a runtime starts, and unexpected communication errors kill and reap its child.

## Isolation and fair comparisons

Fixture copying prevents accidental inclusion of checkers and sibling attempts;
it is **not an operating-system sandbox**. A runtime can still access other files,
global skills, prior memory, the network, and inherited environment. In a serious
A/B study use fresh containers or otherwise restrict visible files to one prepared
task directory, remove globally installed copies of this skill for baseline, and
disable cross-run memory. Run the scorer separately after execution. The checker
executes submitted Python and is for trusted code or an isolated host only.

Use the same model/version, tools, context limits, time budget, and sampling settings
for both conditions. Record those settings outside task inputs. Randomize order
across repeated runs; this minimal runner uses baseline then treatment for each case
and does not implement randomization or statistical significance. Inspect failures
and resource cost, report actual sample size, and avoid treating one paired sweep
as a general agent success rate. Verify traces show the treatment actually read the
skill; access to a skill is not proof of its use.

## Harness self-test and historical results

```bash
python evals/selftest.py
```

This tests fixture isolation, refusal to overwrite, known defects, correct synthetic
repairs, fallback-audit reproducibility, runtime environment/log handling, missing
responses, timeouts, and launch failures. It invokes **no model**. Passing it only
means the harness behaves as tested. Its answer code stays in the test source and
is never copied into prepared tasks.

The earlier [two-scenario behavior record](../evals/results/behavior-checks.json)
is preserved as historical evidence. Its manual prompts and assessment method
differ from version 1 and cannot be combined with new scores as one success rate.

## Recorded actual agent runs

The [v0.2.0 artifact evidence](../evals/results/v0.2.0/README.md) contains four actual
fresh host-agent runs covering two cases, with one baseline and one treatment
attempt each. Both conditions passed all final harness checks. These results do
not demonstrate a skill advantage; see the evidence record for provenance and
its incomplete replay and release-snapshot limitations.
