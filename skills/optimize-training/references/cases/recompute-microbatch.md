# Selective recomputation and microbatching

- **ID:** `recompute-microbatch`
- **Maturity:** `engineering_pattern`
- **Empirical status:** No project measurement supplied; no memory saving or speedup claimed.
- **Track:** System optimization only while the declared update semantics remain valid.

## Applicability and counter-signals

Select this case when memory evidence identifies saved activations as a limiting allocation, or when repeated forward work during backward is a material update cost. Locate the peak by phase; optimizer state, parameter gathering, evaluation, and checkpoint saving can peak elsewhere.

**Counter-signals:** enough memory already exists and recomputation is absent; the dominant occupancy is parameter/optimizer storage; or a communication buffer causes OOM. Do not attribute every large reserved-memory value to fragmentation.

## Mechanism and allowed changes

Treat checkpoint boundaries and microbatch size as separate experiments before testing their interaction. More recomputation may release activation memory at a compute cost; less recomputation may spend existing headroom to remove repeated work. Choose boundaries from module-level evidence, not a universal stride or GPU recipe.

If memory allows a different microbatch, declare it and its accumulation change together. For equal-sized conventional data-parallel microbatches, verify `global_batch = microbatch × data_parallel_replicas × accumulation`; total GPU count is not always replica count. For variable tokens or tail batches, inspect the actual weighted loss denominator. An unchanged product is necessary in this narrow case but does not prove equivalent BatchNorm, stochastic operations, clipping, or gradient synchronization. A changed global batch belongs to a training-policy comparison.

## Correctness checks

Pin the checkpoint implementation and explicit options supported by the installed version. Replay inputs/state, inspect random operations and side effects inside recomputed regions, and compare forward, backward and update behavior. Do not disable RNG preservation just to reduce overhead without a separate semantics decision. Test training/validation transitions and resume where relevant. Run the required task-quality evaluation after numerical checks; neither a non-OOM run nor checkpoint metadata checks prove equivalence.

## Measurement and decision

Measure completed optimizer updates with all accumulation microsteps, not microstep speed alone. Record peak allocated/reserved memory on each device, device-level occupancy, recomputation cost, exposed communication, and repeated useful throughput. Include representative long/large inputs, evaluation and save phases in feasibility checks. Keep only validated combinations whose memory headroom and job cost suit the objective; a lower peak with a slower job may still be useful if feasibility was the declared goal.

## Rollback and interactions

Restore checkpoint boundaries, microbatch, accumulation, and any linked loss-scaling settings as a coherent configuration. Re-measure after compilation, FSDP grouping/prefetch, or precision changes; those can move the peak or alter the value of recomputation. Confirm the rollback can load the intended checkpoint format.

## Sources and version boundary

Engineering workflow: user's v2 guide, sections 2, 8–9. Primary mechanisms: [torch.utils.checkpoint](https://docs.pytorch.org/docs/2.14/checkpoint.html) (re-execution, RNG and implementation constraints) and [CUDA memory semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html) (allocated/reserved distinction). Checked 2026-10-05; these pages identify PyTorch 2.14. Do not mix options from other versions or assume checkpoint implementations have identical behavior.
