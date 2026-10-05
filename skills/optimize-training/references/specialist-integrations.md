# Optional specialist integrations

Use a specialist only when it resolves a concrete question in the core workflow. The core skill remains usable without any external skills. This is a host-agent handoff protocol, not a callable `Skill` API.

## Discover and inspect

Run the discovery tool with explicit local roots. Resolve `<skill-dir>` from this skill's own location. Pass the root of a skill, a directory of skills, or a repository/product directory with a `skills/` child.

```bash
python <skill-dir>/scripts/discover_integrations.py --root /path/to/local/skills --root /path/to/nsight-install --output integrations.json
```

Use `--no-git` to skip optional bounded local Git metadata reads. Use `--registry /path/to/registry.json` for an explicit registry override. The default registry is [integrations.json](../assets/integrations.json).

Write `--output` outside **all** supplied `--root` directories; the example assumes the current directory lies outside both roots. Symlink paths into a root and output aliases of the registry, inspected `SKILL.md` files or declared helpers, including hardlinks, are rejected without modifying inputs. Omit `--output` for stdout. Valid file output is atomic.

The tool reads `SKILL.md` at a root, in direct child directories, and in direct children of `root/skills`. It does not follow symlinks, recurse through unrelated files, scan common home directories, install dependencies, access the network, import torch/pandas, or execute a skill/helper. Repeated roots referring to the same actual file are deduplicated; distinct files with the same frontmatter name are ambiguous. Directory names alone never select a skill.

- `missing`: continue with the relevant bundled reference and core tools. State which specialist analysis was not performed.
- `ambiguous`: do not choose the first match. Use a narrower explicit root from known project configuration or resolve the intended installation with the user only if needed.
- `located`: read the **entire selected `SKILL.md`**, then the specific referenced helper/instructions needed for the task. Check package metadata in the actual training interpreter and verify inputs, versions, permissions and budget before execution. `located` never means runnable.

Record the selected file path, full-content SHA-256, declared version if available, containing Git commit if available, and helper versions/hashes when helpers are subsequently used. Do not infer a version from directory names or the reviewed upstream pin. Missing metadata remains unknown.

## Route by evidence

| Specialist | Give it | Bound its work | Core follow-up |
| --- | --- | --- | --- |
| `profile-model` | Existing profiler JSON/JSON.gz, rank identity, trace units and measured window | Analyze the critical path and supporting events; collect a new trace only within an authorized command and budget | Repeated unprofiled end-to-end measurements; do not sum overlapping kernel durations as wall time |
| `debug-graph-breaks` | Logs/tlparse report, relevant code, framework version, baseline behavior | Diagnose first; modify only allowed paths and reproduce with a bounded command if authorized | Input/output, gradient and optimizer-update checks; warmup/compile costs and end-to-end benefit |
| `analyze-memory-snapshot` | Existing JSON allocator snapshot, run phase, optional compatible second snapshot | Distinguish allocated/reserved/delayed-free memory; read a local helper before executing it | Recheck peak memory and step cost, correctness, and any changed microbatch/global batch protocol |
| `nsight-systems` | Explicit local skill pack and compatible report or capture contract | Follow installed version instructions within the same resource bounds | Relate profiler evidence to the actual training window and validate gains without detailed profiling |

The reviewed `profile-model` DataFrame workflow uses pandas; HTA is optional for deeper analysis. Metadata presence alone proves neither importability nor compatibility. Basic manual inspection of JSON can continue without either. The graph-break skill can review existing logs without torch, but reproduction needs the real runtime, code, data and matching APIs. Existing JSON memory analysis may not need torch/GPU; capture does. A detected `nsys` executable is merely a path, not a checked version or a working installation.

The memory integration preserves the core input-trust contract: **do not unpickle untrusted snapshots**. A user upload or a workspace location does not by itself establish trust. Prefer JSON exported in an already trusted producer environment. Do not execute a helper until its selected input format and loader behavior have been inspected.

## Version and compatibility rules

The registry records reviewed upstream `meta-pytorch/skills` commit `15ea40020f7ab0ad6645f78dca3b5634383e93c9` and the exact SHA-256 and Git blob SHA of each of three `SKILL.md` files. Those pins are provenance, not an automatic dependency lock or runtime approval. No upstream skill code or prompt is vendored.

`pinned_skill_content=match` compares only the installed `SKILL.md` bytes. It does not cover helper scripts, dependencies or behavior. `diverged` warns that the installed entry changed; inspect and record the actual version before deciding compatibility. `unrecorded` means no content pin exists, as for the locally supplied Nsight pack. A containing repository commit may belong to the user's aggregate skill repository; it is not asserted to be the upstream commit. A matching Git HEAD also does not establish an unmodified worktree.

Treat upstream thresholds and examples as task-specific suggestions. Check installed framework documentation before using private or version-sensitive APIs. Do not adopt a generic graph-break reduction, low-level kernel speedup, or allocator setting as evidence that training quality and job performance passed. Existing user authorization takes precedence over an external skill's generic confirmation defaults; it never permits exceeding the authorized action or budget.

## Handoff record

Create this record only when a concrete specialist task is needed. Replace illustrative paths and nulls with observed values; leave execution fields empty until executed. The host may run the bounded workflow itself or delegate it through its actual agent tools. This JSON is a record, not an executable runner configuration.

```json
{
  "schema_version": 1,
  "specialist": "profile-model",
  "selected_skill_file": "/observed/local/profile-model/SKILL.md",
  "skill_sha256": null,
  "full_instructions_read": false,
  "question": "What explains idle time in the measured training window?",
  "inputs": [{"path": "/observed/run/trace.json", "sha256": null, "rank": 0}],
  "measurement_contract": "/observed/run/experiment.json",
  "scope": {
    "mode": "existing_artifacts_only",
    "allowed_write_paths": ["/observed/run/analysis/"],
    "allowed_training_changes": [],
    "allowed_commands": [],
    "max_wall_seconds": null,
    "max_gpu_seconds": 0
  },
  "deliverables": ["analysis.json", "evidence.md"],
  "required_return_fields": ["finding", "evidence_paths", "commands", "exit_codes", "limitations", "proposed_changes", "checks_not_run"],
  "status": "planned",
  "execution": [],
  "core_acceptance": "not_run"
}
```

Use absolute artifact paths and keep raw traces local to their authorized destination. In a delegated prompt, give the selected skill path, the question, inputs, scope/budget and required return fields; require the delegate to read the full skill and return actual evidence. Preserve `planned`, `executed`, `passed`, `failed`, and `not_run` distinctions at the individual check level. The coordinator owns final acceptance and must review the returned artifacts; a specialist's summary cannot promote core gates to passed.

## Fallback without a specialist

Read [profiling.md](profiling.md), [compile-runtime.md](compile-runtime.md) or [distributed-memory.md](distributed-memory.md) for the relevant track. Finish environment inventory, code review, sample coverage, tensor comparisons and measurement planning that current evidence permits. If a helper/runtime/input is absent, report that exact unrun analysis rather than fabricating outputs or blocking independent core work.
