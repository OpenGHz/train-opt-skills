# Repair duplicated rank/worker shards

- **ID:** `shard-repair`
- **Maturity:** `engineering_pattern`
- **Empirical status:** No project measurement supplied; no training-quality or throughput gain claimed.
- **Track:** Correctness repair. Establish the repaired version as the subsequent optimization baseline.

## Applicability and counter-signals

Select this case when actual emitted sample identities show repetition inconsistent with the sampling contract, despite different sampler indices or busy devices. Trace an index through the dataset: an internal cursor can ignore it, and iterable replicas may each read the same source.

**Counter-signals:** repeats are required by a replacement sampler, documented tail padding, or the training schedule; identities omit dataset version/episode/time and falsely collide; or the audit window combines epochs unintentionally. Diagnose against expected multiplicities, not a universal no-repeat rule.

## Mechanism and allowed changes

Make the actual reader honor one ownership scheme across data-parallel ranks and loading workers. Use the data-parallel sampling space, not automatically the global process group. Pass validated rank metadata to worker datasets; do not silently substitute rank zero when distributed initialization was required.

For a common ordered chunk list and equal worker counts, a rank/worker stride assignment is a candidate; it is not a proof about chunk contents. Avoid double-sharding an already partitioned source. Handle zero subprocess workers, empty shards, filtering, uneven workloads, tail behavior, and synchronized termination. Elastic membership or unequal worker counts needs a separately specified ownership/resume protocol.

Preserve intended data membership, augmentation, and optimizer policy. The repair changes the defective realized sample stream to the intended stream; label that explicitly instead of claiming equivalence to the buggy baseline.

## Correctness checks

Audit emitted IDs with epoch, rank, worker, update and dataset identity over a finite known fixture. Check expected membership and multiplicities after filtering/tail rules, including chunks fewer than consumers. Check ordering/shuffling between epochs and continuation from consumed progress; prefetched reader progress may be ahead of training consumption. Validate collective termination and updates when ranks receive unequal work. Distinguish planned checks from checks actually run.

## Measurement and decision

Report repair evidence first. Separate processed entries, unique identities, and unintended repeats; unique identities do not imply statistical independence. A change in entry throughput against the defective reader is descriptive, not an equivalent-training speedup. Re-establish real-data timing and quality baselines after the repair, then evaluate system optimizations. If the expected set is unavailable, state that duplicate detection does not certify complete coverage.

## Rollback and interactions

Keep the previous implementation/config for diagnosis, but do not resume known-defective production training as the default rollback. Stop the affected run or restore a separately validated reader/checkpoint. Record whether existing checkpoints consumed invalid data. Recheck ownership after worker-count, sampler, prefetch, or elastic-world-size changes.

## Sources and version boundary

Engineering workflow: user's v2 guide, sections 5.3–5.4. Primary mechanisms: [PyTorch data loading](https://docs.pytorch.org/docs/2.14/data.html) (iterable worker replicas, sampler padding and `set_epoch`). Checked 2026-10-05; the page identifies PyTorch 2.14. Inspect the actual dataset/framework wrapper and version rather than assuming a sampler controls its internal reader.
