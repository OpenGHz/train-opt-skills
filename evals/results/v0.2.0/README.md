# v0.2.0 evaluation evidence

This directory preserves artifacts from **four actual fresh delegated host agent
sessions**: one baseline and one treatment attempt for each of two cases. These are
not the synthetic test doubles used by `evals/selftest.py`.

| Case | Baseline | Treatment | Trials per condition |
|---|---|---|---:|
| Per-image normalization (`augmentation_stats`) | Passed 5/5 final harness checks | Passed 5/5 final harness checks | 1 |
| Same-shape QKV reorder (`weight_reorder`) | Passed 5/5 final harness checks | Passed 5/5 final harness checks | 1 |

**Both conditions passed. These runs do not demonstrate a skill advantage.** No
GPU, throughput, training-time, or production-training result was measured. The
other four suite cases were not part of this paired execution.

## Execution provenance

The host prepared separate fixture directories and delegated each task to a fresh
agent session. Treatment received the core skill; baseline received the task
fixtures. The exact model identifier was not exposed to the recording agent and
is recorded as `null`; no model version, token usage, or duration is inferred.
The harness `execute()` runtime adapter was **not** used. All four completed
workspaces were scored using the final deterministic harness.

Preparation manifests are preserved byte-for-byte. Their `execution: "not_run"`
field reflects the original preparation step, which the separate host sessions did
not update. Actual manual execution provenance is recorded in this README and
[summary.json](summary.json); the preserved responses, repaired code, supporting
artifacts, and final score reports provide the corresponding artifact evidence.

Some final release integration hardening followed these trials. They therefore do
not constitute an end-to-end evaluation of every component in the final release.

## Included artifacts and limits

Each attempt contains its original top-level regular files: task prompt, response
schema, repaired code, response, and any verification scripts, recorded results,
patches, or reports created by that agent. `metadata/` contains the original
preparation manifest and final score. The summary lists every bundled attempt file
and its SHA-256 digest. Caches and the copied `_skill/` directory are excluded.

Input and treatment skill snapshot hashes remain in the manifests. The exact full
skill snapshot and complete agent transcripts/tool traces are **not** bundled, so
this is a compact artifact record, not a complete execution replay. Agent-authored
verification claims can be inspected alongside their scripts and saved results;
the separately recorded harness score covers only its specified deterministic
checks. Some agent verification scripts use NumPy, while the harness itself uses
only the Python standard library.

A single trial per condition on two small tasks is insufficient for a general
success-rate estimate or statistical comparison. Baseline may also see host-level
capabilities outside fixture copying; the [evaluation guide](../../../docs/evaluation.md)
explains why stronger isolation is needed for a controlled study. Archived prompts
retain their original `_skill/` path references, but that snapshot is not present in
this compact bundle. Prepare fresh workspaces to conduct new runs.
