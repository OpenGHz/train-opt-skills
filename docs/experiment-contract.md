# Experiment contract v2

An implementation optimization may change execution settings while preserving the
training task and the measurement method. The v2 benchmark record makes this
distinction explicit. Passing the comparison means the *declared* contract is
consistent; it does not establish training correctness or convergence.

The tool remains dependency-free. Version 1 records continue to require exact
equality of `measurement` and `protocol`, including additional fields. Version 1
and version 2 cannot be mixed in a comparison.

## Two different artifacts

- [`assets/experiment.json`](../skills/optimize-training/assets/experiment.json)
  is a **planning template**, marked `experiment_schema_version: 2`. Fill the
  placeholders and preserve plans, acceptance criteria, evidence and decisions.
  It is not a benchmark input and should not be passed to the comparison tool.
- [`controlled-baseline.json`](../examples/controlled-baseline.json) and
  [`controlled-candidate.json`](../examples/controlled-candidate.json) are
  **completed synthetic benchmark records**, marked `schema_version: 2`.
  Copy these shapes when exporting actual measurements.

```bash
python skills/optimize-training/scripts/compare_benchmarks.py \
  examples/controlled-baseline.json examples/controlled-candidate.json
```

These fixtures contain invented timings for tool demonstration; they provide no
evidence that compilation or additional data workers improve a real workload.

## Benchmark record sections

Each record requires all eight root fields shown below; other root fields are
rejected so that a setting cannot accidentally be placed outside its contract.

| Field | Meaning and comparison rule |
|---|---|
| `schema_version` | Integer `2`. |
| `synthetic` | Boolean; use `false` only for real measurements. Either input marked synthetic adds an explicit warning. |
| `comparison` | Contains `track` and `changed_execution_settings` only. Supported track: `system_optimization`. Both records must declare the same exact set of changed settings. |
| `measurement` | The entire object must match, including any extra fields. |
| `invariants` | The entire object must match, including any extra fields. This describes task semantics and the fixed environment. |
| `execution` | Explicit values for implementation settings. Both records must contain the same keys, and actual changed keys must equal the declared set. |
| `provenance` | Requires a nonempty `code_revision`; may include commands and artifact identifiers. Differences are retained in the report and are not automatically treated as semantic changes. |
| `runs` | Nonempty list of measured windows, each with positive `elapsed_seconds` and `effective_units`; optional positive `update_times_ms` list. |

Settings belong in `invariants` or `execution`, even if a complete command is also
recorded in provenance. Do not hide meaningful configuration differences in a
revision string, command, or a generic mode label. No metadata tool can verify
that a declaration truthfully describes executed code; preserve configuration
files, the implementation diff and raw logs alongside each benchmark.

### Measurement is always fixed

Required fields are the same as v1:

- Nonempty strings: `unit`, `scope`, `synchronization`.
- `warmup_excluded` must be `true`.
- Booleans: `includes_data_loading`, `includes_validation`,
  `includes_checkpointing`.

Record the actual warmup procedure, window definition, repeats, device/rank
synchronization method and job boundary as extra fields. The tool compares them
as fixed data. `changed_execution_settings` cannot exempt a measurement field.
For distributed runs, count completed effective work across data-parallel replicas
once, without multiplying it by tensor/pipeline-parallel replicas. Preserve any
intentional resampling in the agreed protocol; audit distinct sample coverage
separately. Elapsed time must include the slowest participating rank.

### Invariants are always fixed

Required nonempty strings: `workload_id`, `dataset_id`, `model_id`, `precision`,
`input_shape`, `loss`, `optimizer`, `sampling_and_augmentation`, `initial_weights`,
`hardware`, `software`. Also required: positive integers `global_batch_size` and
`views`, and integer `seed`.

Use a per-example `input_shape`, excluding the microbatch dimension. Include
resolution, sequence lengths and padding semantics where applicable. Identify
dataset version/split/order, model configuration, precision and TF32 policy,
loss reduction, optimizer hyperparameters/scheduler, augmentation coupling and
initial checkpoint unambiguously, preferably with immutable config identifiers.
Additional invariant fields are allowed and are compared strictly.

Reducing resolution or views, changing the dataset, loss, global batch, or precision
cannot be declared an equivalent execution change. Those may be worthwhile
experiments, but require a separate task-quality study. New software/hardware
environments are also outside this first version's controlled comparison.

### Execution settings and exact declarations

The schema deliberately has a closed set of setting names. Unrecognized fields,
arbitrary nested objects and duplicate declaration names are rejected. Required:

| Setting | Type | Rule |
|---|---|---|
| `microbatch_size` | Positive integer | Samples per data-parallel replica per microstep. May change if declared. |
| `gradient_accumulation_steps` | Positive integer | May change if declared; verify update cadence separately. |
| `world_size` | Positive integer | Fixed. Total participating processes. |
| `data_parallel_size` | Positive integer | Fixed. Must divide `world_size`; use this factor for aggregate batch, not total tensor/pipeline parallel processes. |
| `compile` | Nonempty string | Explicit implementation mode, such as `disabled` or `default`. |
| `num_workers` | Nonnegative integer | Explicit data-loader worker count. |

Optional supported settings, supplied on **both** sides when used:

| Type | Settings |
|---|---|
| Boolean | `activation_checkpointing`, `pin_memory`, `non_blocking_transfer`, `persistent_workers`, `fsdp_reshard_after_forward` |
| Positive integer | `prefetch_factor` |
| Positive number | `ddp_bucket_cap_mb` |
| Nonempty string | `memory_format`, `optimizer_implementation`, `attention_implementation` |

`persistent_workers: true` and a supplied `prefetch_factor` require
`num_workers > 0`. Omit `prefetch_factor` on both sides when either side uses
zero workers; record implementation-specific conditional prefetch behavior in
the experiment notes. Changing optimizer or attention implementations is only
eligible when the optimizer/loss/model/precision contracts remain fixed. The
tool does not prove equivalence of these implementations.

For example, both records declare:

```json
{
  "track": "system_optimization",
  "changed_execution_settings": ["compile", "num_workers"]
}
```

The candidate must change both listed values, and no other execution value.
A declaration is an exact list of this experiment's changes, not a permission
list for future experiments. Adding a setting on only one side is incomparable:
record its old and new values explicitly. Declaration order does not matter.

### Microbatch changes are conditional, not a proof of equivalence

Each record must satisfy:

```text
global_batch_size = microbatch_size × gradient_accumulation_steps × data_parallel_size
```

For example, at data-parallel size 2 and global batch 32, microbatch/accumulation
`16 × 1` may be compared with `8 × 2` after declaring both changes. The tool checks
the arithmetic and preserves `correctness: not_assessed`. Audit batch-dependent
layers, loss normalization, RNG behavior, synchronization, optimizer and scheduler
cadence, gradients and actual parameter updates. Equal aggregate batch does not
make all training implementations equivalent. Unequal replica batches, variable
microbatch schedules and partial accumulation windows need a richer contract
and are not representable by this simple product schema.

## Results and non-supported comparisons

| Exit | Meaning |
|---|---|
| `0` | Comparable declarations. Reports throughput and its ratio; correctness remains unassessed. |
| `1` | Incomparable: changed fixed contract, undeclared/unperformed changes, changed topology, mixed schema versions, or unsupported track. No `speedup` is emitted. |
| `2` | Malformed input, including missing fields, prohibited change names, wrong types, or inconsistent aggregate batch. |

`task_change` and `scaling` are recognized track labels, but this tool returns
`incomparable` with `unsupported_comparison_track`. It does not convert those
studies into equivalent-work speedups. Use separate task-quality or strong/weak
scaling analyses with explicit work and resource definitions.

Throughput is `sum(effective_units) / sum(elapsed_seconds)`. The report's elapsed
time reduction is implied for equal effective work; it is not a claim about full
job duration or time to target quality. A changed revision alone is permitted,
but review its diff and validate the invariant declarations. A comparable result
is evidence of measurement-contract consistency only; promotion still requires
the experiment's independent correctness and task-quality evidence.
