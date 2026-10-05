# DataLoader supply on the exposed path

- **ID:** `dataloader-supply`
- **Maturity:** `engineering_pattern`
- **Empirical status:** No project measurement supplied; no speedup claimed.
- **Track:** System optimization, provided sampling and preprocessing remain within contract.

## Applicability and counter-signals

Select this case when a representative trace connects device idle gaps to waiting for real batches, and phase measurements identify reading, decoding, augmentation, collation, or interprocess transfer. A resident synthetic-data comparison can isolate supply costs, but cannot replace real-data acceptance.

**Counter-signals:** the next batch is ready while communication blocks progress; all host cores or storage are saturated; a slow rank has longer model inputs; or logging causes the gaps. More workers may then increase contention without fixing the cause.

## Mechanism and allowed changes

Test one supply change at a time: worker count, supported prefetch configuration, worker persistence, deterministic decoding cache, or storage layout. Budget all training ranks' workers together. Check option validity when `num_workers=0`; do not assume multiprocessing-only options apply.

Keep dataset version, split, effective input content, sampling policy, augmentation semantics, and update composition fixed unless the contract explicitly declares otherwise. Cache only deterministic results with a versioned key. Do not cache random augmentation outputs or drop a real view/resolution to make loading look faster; those alter the training protocol.

## Correctness checks

Audit actual sample IDs across rank/worker, epochs, tails, and resume. Replay a saved set of augmentation parameters to compare final model inputs before and after the change; separately check random sampling policy. Equal seeds across changed worker counts do not prove the same draws. Verify a representative update from identical inputs and state. Test persistent-worker epoch behavior and recovery progress if affected.

## Measurement and decision

Record the baseline and candidate worker/prefetch settings, total CPU budget, host and pinned-memory peaks, storage traffic, real-data wait time, and repeated completed-update throughput. Count useful work using the same policy, include every accumulation microstep, and synchronize only measurement boundaries as needed. Report startup/cache preparation separately from steady state and include it in job cost. Keep the candidate only after required checks and sustainable end-to-end improvement; a shorter isolated decode benchmark is insufficient.

## Rollback and interactions

Restore the prior loader configuration and cache selection; preserve the failed run's evidence. Remove only newly created disposable cache artifacts when appropriate. Recheck shard ownership when adding workers, CPU pressure after GPU-side preprocessing, and host-buffer lifetime if transfer overlap changes. Refer to [shard repair](shard-repair.md) if the audit finds duplication.

## Sources and version boundary

Engineering workflow: user's v2 guide, sections 4–6. Primary mechanisms: [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) (worker-based overlap) and [DataLoader documentation](https://docs.pytorch.org/docs/2.14/data.html) (worker replicas/options). Checked 2026-10-05; the latter page identifies PyTorch 2.14. Use documentation matching the installed PyTorch version before selecting options.
