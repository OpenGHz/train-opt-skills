# Training optimization decision cases

Use these original engineering patterns to select a hypothesis and its checks. They are **not measured speedup reports**. `engineering_pattern` means a proposed reusable method; `synthetic_counterexample` means an intentionally constructed failure. Neither label establishes effectiveness on the current project.

Read this index, then read only the linked case matching observed evidence. From the skill directory, search with `rg -n 'dataloader|compile|recompute|shard|augmentation' references/optimization-casebook.md`; search within the selected case for `Counter-signals`, `Correctness`, or `Rollback` when needed.

| Evidence or question | Case | Maturity | Intended decision |
| --- | --- | --- | --- |
| Device gaps coincide with real-data retrieval or decoding | [DataLoader supply](cases/dataloader-supply.md) | `engineering_pattern` | Test supply concurrency only where waits are exposed |
| Hot regions exit compilation or input shapes cause repeated compilation | [Compile boundaries and shapes](cases/compile-shapes.md) | `engineering_pattern` | Choose a validated compilation boundary and shape policy |
| Activation peaks constrain batching or recomputation dominates updates | [Selective recomputation and microbatching](cases/recompute-microbatch.md) | `engineering_pattern` | Evaluate memory and time jointly |
| Busy ranks consume unexpectedly repeated sample identities | [Repair duplicated shards](cases/shard-repair.md) | `engineering_pattern` | Establish a correct sampling baseline before optimization |
| Vectorized augmentation changes an image when its batch neighbors change | [Failed augmentation candidate](cases/augmentation-batch-statistics.md) | `synthetic_counterexample` | Reject the candidate and restore the intended statistics |

## Use a case without turning it into a recipe

1. Match evidence and counter-signals. A low utilization percentage alone does not select a case.
2. Classify the work as correctness repair, system optimization, or training/task change. Changing real camera views, effective image resolution, sequence content, or global batch needs the corresponding task-quality contract; it is not made equivalent by a favorable throughput number.
3. Write invariants and declared changes before running the experiment. Freeze actual comparison inputs, weights, noise, and augmentation parameters where possible. Equal seeds are not proof of equivalent random draws.
4. Apply the smallest candidate, run its required correctness checks, and measure a representative workload. A graph without breaks, successful state loading, or a passed tensor check alone cannot certify final task quality.
5. Record keep/revert/inconclusive/blocked with evidence. Do not promote a pattern to a measured result using hypothetical numbers, CPU fixtures, or a benchmark from unrelated hardware.

## Record a reusable observation only when useful

Copy [the case record](../assets/optimization-case.json) into the task's artifact directory when a durable decision case would help the user. Routine task reports do not require modifying this skill or writing to a shared knowledge base. Curating or publishing a case is a separate, authorized maintenance action.

Keep provenance, revision and environment fingerprints; applicability and counter-signals; unchanged invariants and declared changes; raw artifacts and exact measurement boundaries; required checks with `planned`/`not_run`/`passed`/`failed`/`not_applicable` and evidence; quality evidence; rollback; interactions; and unresolved limits. Explain every `not_applicable`. Record empirical measurements only when they exist. Use `measured_case` only after actual execution is attached, and state whether measurement covers a microbenchmark, CPU fixture, GPU window, or full training job.

The cases were derived from the user's *Training Performance Optimization Guide v2* (2026-10-05), especially sections 2–9, and checked against the primary sources linked in each case. The decision workflows and counterexample are original project material; no upstream performance claim or GPU configuration is imported. Official `stable`, `main`, and tutorial pages move: resolve APIs against the project's locked version and record that version and access date.
