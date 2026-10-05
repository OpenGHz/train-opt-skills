# Optional specialist skills

Train Opt Skills owns the training contract and final acceptance. Optional skills provide bounded diagnostic work. The package includes an offline discovery tool, reviewed source pins and a handoff protocol; it does not install or automatically execute upstream skills.

## Discover local installations

From this project's root, supply directories that you actually want inspected:

```bash
python skills/optimize-training/scripts/discover_integrations.py \
  --root /absolute/path/to/installed-skills \
  --root /absolute/path/to/nsight-installation \
  --output integrations.json
```

`--root` is repeatable and required. Each root may be a single skill directory, a directory of skills or a repository/product directory containing `skills/`. Discovery is shallow and ignores symlinks. No filesystem-wide search, network request, package import, helper execution or installation occurs. `--no-git` disables optional local Git revision reads. `--registry` accepts a replacement JSON registry; by default the bundled [registry](../skills/optimize-training/assets/integrations.json) is used.

Place `--output` outside **every** supplied `--root` directory; the example assumes the current directory is outside both roots. The tool also rejects symlink paths into those roots and output aliases of the registry, inspected `SKILL.md` files or declared helpers, including hardlinks. Omit `--output` to print the report to stdout. A rejected output leaves inputs unchanged; valid reports are written atomically.

| Integration | Appropriate input | Dependency boundary |
| --- | --- | --- |
| `profile-model` | Existing PyTorch JSON/JSON.gz traces, including rank-local traces | pandas for the reviewed DataFrame workflow; HTA optional; basic JSON inspection can proceed without them |
| `debug-graph-breaks` | Existing graph-break logs and matching user code | torch and the actual training runtime are needed for reproduction, not for static review |
| `analyze-memory-snapshot` | Existing JSON allocator snapshot, optionally a comparable pair | Existing JSON review may work without torch; capture requires a compatible runtime; a separately installed helper is detected but not run |
| `nsight-systems` | Explicit local product skill pack | Product/version compatibility must be checked; a detected `nsys` path is not runtime validation |

All integrations are optional. The discovery report describes prerequisites and metadata availability, not whether the specialist is fully runnable. Package versions belong to the interpreter running discovery; rerun discovery in the training interpreter when those differ.

## Understand the report

The top-level JSON contains `schema_version`, `mode`, `roots`, `registry_revision`, `integrations`, `warnings` and `limits`. Each integration contains:

| Field | Meaning |
| --- | --- |
| `status` | `missing`, `located` (one file) or `ambiguous` (multiple different files with the same frontmatter name) |
| `selected_skill_file` | Absolute entry path only when exactly one match exists |
| `matches` | Actual directory/file, SHA-256 of the complete entry, declared version or unknown, optional containing repository commit, detected helpers |
| `upstream` | Reviewed source location and file/commit pins; not the installed version |
| `dependencies` | Distribution metadata presence/version and its conditional purpose; no imports are tested |
| `executables` | Optional executable path inventory; nothing was executed |
| `artifact_requirements` | Task inputs still requiring validation by the host agent |
| `capability_verified` | Always `false` for this inventory-only tool |

Duplicate roots reaching the same actual file are deduplicated. Copies at different locations remain ambiguous even when their contents match. A directory named `profile-model` with a different frontmatter name is not accepted as that skill. Generated directory names are fine when their frontmatter matches. A malformed/missing root is recorded in `warnings`; missing optional integrations do not fail the CLI. Invalid registry/JSON or an unwritable output yields exit code `2`; a completed inventory yields `0`.

The frontmatter reader intentionally accepts literal scalar `name` and optional nested `metadata.version`, not arbitrary YAML execution or complex YAML aliases. An unrecognized representation is reported rather than guessed.

## Provenance and limitations

The three meta-pytorch entry files were fetched and inspected at commit [`15ea40020f7ab0ad6645f78dca3b5634383e93c9`](https://github.com/meta-pytorch/skills/tree/15ea40020f7ab0ad6645f78dca3b5634383e93c9). Their exact UTF-8 SHA-256 values and Git blob hashes are recorded separately in the registry. No upstream prompts or helper implementations are copied into this project. They remain separately installed, independently licensed components.

An installed entry gets `pinned_skill_content=match`, `diverged` or `unrecorded`. A match covers only `SKILL.md`; it says nothing about helper modifications or runtime dependencies. Divergence requires renewed inspection, not automatically rejecting an intentional local customization. The containing Git commit is recorded without reading remote URLs; it may be an aggregate skills repository and does not establish origin or a clean worktree. Nsight has no asserted file/version pin because its actual installed product version is authoritative.

## Use the handoff

Read the selected skill in full, then check only the referenced resources needed for the task. Use the [host-agent handoff record](../skills/optimize-training/references/specialist-integrations.md#handoff-record) to identify actual input artifacts, the task question, allowed edits, execution budget and expected output evidence. Invoke the real host's delegation tools or perform the work in the current agent; this package invents no cross-agent `Skill` API.

If no unique installation exists, continue with core references and tools and state which specialist analysis remains unrun. If a specialist runs, record its commands, exits, artifacts, limitations and observed checks. The core workflow then rechecks correctness and repeated end-to-end performance. Eliminated graph breaks, a plausible snapshot diagnosis or a profiler recommendation alone cannot pass training acceptance.

The core input-trust rule remains in force: do not unpickle an untrusted file because it was uploaded or found locally. Prefer JSON exported in a trusted producer runtime. Inspect any installed helper's loader behavior before use. Respect the user's established authorization and resource budget when adapting specialist instructions.
