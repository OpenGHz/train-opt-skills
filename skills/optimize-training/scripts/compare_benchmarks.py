#!/usr/bin/env python3
"""Compare repeated benchmark windows; this never verifies training correctness.

Version 1 JSON: schema_version=1, synthetic=boolean, measurement={unit, scope,
warmup_excluded:true, synchronization, includes_data_loading:boolean,
includes_validation:boolean, includes_checkpointing:boolean}, protocol={workload_id,
dataset_id, model_id, precision, global_batch_size:positive-int, world_size:positive-int,
gradient_accumulation_steps:positive-int, input_shape, hardware, software, seed:integer},
runs=[{elapsed_seconds:positive-number,effective_units:positive-number,
update_times_ms?:[positive-number,...]}, ...]. Unmarked fields are nonempty strings.
Measurement and protocol dictionaries (including additional fields) must match exactly.
Version 2 replaces protocol with invariant, execution and provenance sections:
invariants and measurement must match exactly; execution differences must match
comparison.changed_execution_settings exactly. See docs/experiment-contract.md.
Synthetic inputs are useful for tooling validation, not evidence of training speedup.
Exit: 0 comparable; 1 incomparable measurement/protocol; 2 malformed input.
"""
from __future__ import annotations
import argparse
import math
import json
import statistics
import sys
from _common import InvalidInput, emit, integer, load_json, nonempty, positive, protect_output


# Deliberately closed: an arbitrary setting must not smuggle a task/precision
# change into the list of permitted implementation changes. Extend this schema
# explicitly when a new execution mechanism has a reviewed semantic boundary.
EXECUTION_TYPES = {
    "microbatch_size": "positive_int", "gradient_accumulation_steps": "positive_int",
    "world_size": "positive_int", "data_parallel_size": "positive_int",
    "compile": "string", "num_workers": "nonnegative_int",
    "activation_checkpointing": "bool", "pin_memory": "bool",
    "non_blocking_transfer": "bool", "persistent_workers": "bool",
    "prefetch_factor": "positive_int", "memory_format": "string",
    "optimizer_implementation": "string", "attention_implementation": "string",
    "ddp_bucket_cap_mb": "positive_number", "fsdp_reshard_after_forward": "bool",
}
REQUIRED_EXECUTION = {
    "microbatch_size", "gradient_accumulation_steps", "world_size",
    "data_parallel_size", "compile", "num_workers",
}
FIXED_EXECUTION = {"world_size", "data_parallel_size"}


def same(left, right):
    """Keep JSON types distinct (e.g. true is not the number 1)."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def validate_v2(record, label):
    allowed = {"schema_version", "synthetic", "comparison", "measurement", "invariants",
               "execution", "provenance", "runs"}
    unknown = set(record) - allowed
    if unknown:
        raise InvalidInput(f"{label}: unknown v2 root fields: {sorted(unknown)}")
    comparison = record.get("comparison")
    if not isinstance(comparison, dict) or set(comparison) != {"track", "changed_execution_settings"}:
        raise InvalidInput(f"{label}: comparison must contain track and changed_execution_settings only")
    if comparison["track"] not in ("system_optimization", "task_change", "scaling"):
        raise InvalidInput(f"{label}: unknown comparison track")
    changed = comparison["changed_execution_settings"]
    if not isinstance(changed, list) or any(not isinstance(key, str) or not key.strip() for key in changed):
        raise InvalidInput(f"{label}: changed_execution_settings must be a list of setting names")
    if len(set(changed)) != len(changed):
        raise InvalidInput(f"{label}: changed_execution_settings contains duplicate names")
    unsupported = set(changed) - (set(EXECUTION_TYPES) - FIXED_EXECUTION)
    if unsupported and comparison["track"] == "system_optimization":
        raise InvalidInput(f"{label}: settings cannot be declared as equivalent execution changes: {sorted(unsupported)}")
    invariants = record.get("invariants")
    if not isinstance(invariants, dict):
        raise InvalidInput(f"{label}: invariants object is required")
    for field in ("workload_id", "dataset_id", "model_id", "precision", "input_shape",
                  "loss", "optimizer", "sampling_and_augmentation", "initial_weights",
                  "hardware", "software"):
        nonempty(invariants.get(field), f"{label}.invariants.{field}")
    for field in ("global_batch_size", "views"):
        integer(invariants.get(field), f"{label}.invariants.{field}", 1)
    integer(invariants.get("seed"), f"{label}.invariants.seed")
    execution = record.get("execution")
    if not isinstance(execution, dict):
        raise InvalidInput(f"{label}: execution object is required")
    missing, extra = REQUIRED_EXECUTION - set(execution), set(execution) - set(EXECUTION_TYPES)
    if missing or extra:
        raise InvalidInput(f"{label}: execution missing fields {sorted(missing)}; unknown fields {sorted(extra)}")
    for field, value in execution.items():
        kind = EXECUTION_TYPES[field]
        path = f"{label}.execution.{field}"
        if kind in ("positive_int", "nonnegative_int"):
            integer(value, path, 1 if kind == "positive_int" else 0)
        elif kind == "positive_number":
            positive(value, path)
        elif kind == "bool":
            if type(value) is not bool:
                raise InvalidInput(f"{path} must be boolean")
        else:
            nonempty(value, path)
    dp, world = execution["data_parallel_size"], execution["world_size"]
    if world % dp:
        raise InvalidInput(f"{label}: world_size must be divisible by data_parallel_size")
    batch = execution["microbatch_size"] * execution["gradient_accumulation_steps"] * dp
    if batch != invariants["global_batch_size"]:
        raise InvalidInput(f"{label}: global_batch_size must equal microbatch_size * gradient_accumulation_steps * data_parallel_size")
    if execution.get("persistent_workers") is True and execution["num_workers"] == 0:
        raise InvalidInput(f"{label}: persistent_workers requires num_workers > 0")
    if "prefetch_factor" in execution and execution["num_workers"] == 0:
        raise InvalidInput(f"{label}: prefetch_factor requires num_workers > 0; omit it in both records otherwise")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise InvalidInput(f"{label}: provenance object is required")
    nonempty(provenance.get("code_revision"), f"{label}.provenance.code_revision")


def validate(record, label):
    if not isinstance(record, dict):
        raise InvalidInput(f"{label}: root must be an object")
    if type(record.get("schema_version")) is not int or record["schema_version"] not in (1, 2):
        raise InvalidInput(f"{label}: schema_version must be 1 or 2")
    if type(record.get("synthetic")) is not bool:
        raise InvalidInput(f"{label}: synthetic must be boolean")
    measurement = record.get("measurement")
    if not isinstance(measurement, dict):
        raise InvalidInput(f"{label}: measurement object is required")
    for field in ("unit", "scope", "synchronization"):
        nonempty(measurement.get(field), f"{label}.measurement.{field}")
    if measurement.get("warmup_excluded") is not True:
        raise InvalidInput(f"{label}: warmup_excluded must be true; separate warmup first")
    for field in ("includes_data_loading", "includes_validation", "includes_checkpointing"):
        if type(measurement.get(field)) is not bool:
            raise InvalidInput(f"{label}.measurement.{field} must be boolean")
    if record["schema_version"] == 1:
        protocol = record.get("protocol")
        if not isinstance(protocol, dict):
            raise InvalidInput(f"{label}: protocol object is required")
        for field in ("workload_id", "dataset_id", "model_id", "precision", "input_shape", "hardware", "software"):
            nonempty(protocol.get(field), f"{label}.protocol.{field}")
        for field in ("global_batch_size", "world_size", "gradient_accumulation_steps"):
            integer(protocol.get(field), f"{label}.protocol.{field}", 1)
        integer(protocol.get("seed"), f"{label}.protocol.seed")
    else:
        validate_v2(record, label)
    runs = record.get("runs")
    if not isinstance(runs, list) or not runs:
        raise InvalidInput(f"{label}: runs must be a nonempty list")
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise InvalidInput(f"{label}.runs[{index}] must be an object")
        for field in ("elapsed_seconds", "effective_units"):
            positive(run.get(field), f"{label}.runs[{index}].{field}")
        if "update_times_ms" in run:
            times = run["update_times_ms"]
            if not isinstance(times, list) or not times:
                raise InvalidInput(f"{label}: update_times_ms must be a nonempty list if present")
            for value in times:
                positive(value, f"{label}.update_times_ms")
    return record


def summarize(record):
    runs = record["runs"]
    duration = math.fsum(run["elapsed_seconds"] for run in runs)
    units = math.fsum(run["effective_units"] for run in runs)
    throughputs = [run["effective_units"] / run["elapsed_seconds"] for run in runs]
    rate = units / duration
    if not all(math.isfinite(x) and x > 0 for x in [duration, units, rate] + throughputs):
        raise InvalidInput("Benchmark values overflowed or underflowed; use a reasonable measurement scale")
    times = [value for run in runs for value in run.get("update_times_ms", [])]
    result = {"run_count":len(runs), "total_elapsed_seconds":duration,
              "total_effective_units":units, "throughput":rate,
              "median_run_throughput":statistics.median(throughputs),
              "min_run_throughput":min(throughputs), "max_run_throughput":max(throughputs)}
    if times:
        result["pooled_median_update_time_ms"] = statistics.median(times)
        result["update_time_samples"] = len(times)
    return result


def compare(baseline, candidate):
    validate(baseline, "baseline")
    validate(candidate, "candidate")
    if baseline["schema_version"] != candidate["schema_version"]:
        return {"status":"incomparable", "mismatched_sections":["schema_version"],
                "correctness":"not_assessed"}, 1
    version = baseline["schema_version"]
    sections = ("measurement", "protocol") if version == 1 else ("measurement", "invariants")
    differences = [key for key in sections if not same(baseline[key], candidate[key])]
    if version == 2:
        tracks = [record["comparison"]["track"] for record in (baseline, candidate)]
        if tracks != ["system_optimization", "system_optimization"]:
            return {"status":"incomparable", "reason":"unsupported_comparison_track",
                    "tracks":tracks, "correctness":"not_assessed",
                    "explanation":"Task-change and scaling studies need a separate evaluation contract; no equivalent-work speedup is reported."}, 1
        if differences:
            return {"status":"incomparable", "reason":"fixed_contract_changed",
                    "mismatched_sections":differences, "correctness":"not_assessed"}, 1
        left, right = baseline["execution"], candidate["execution"]
        if set(left) != set(right):
            return {"status":"incomparable", "reason":"execution_fields_differ",
                    "explanation":"Record explicit baseline and candidate values for the same settings.",
                    "correctness":"not_assessed"}, 1
        changed = {key for key in left if not same(left[key], right[key])}
        if changed & FIXED_EXECUTION:
            return {"status":"incomparable", "reason":"parallel_topology_changed",
                    "changed_execution_settings":sorted(changed), "correctness":"not_assessed"}, 1
        declared_before = set(baseline["comparison"]["changed_execution_settings"])
        declared_after = set(candidate["comparison"]["changed_execution_settings"])
        if declared_before != declared_after or changed != declared_before:
            return {"status":"incomparable", "reason":"declaration_mismatch",
                    "actual_changed_execution_settings":sorted(changed),
                    "baseline_declared_changes":sorted(declared_before),
                    "candidate_declared_changes":sorted(declared_after),
                    "correctness":"not_assessed"}, 1
    if differences:
        return {"status":"incomparable", "mismatched_sections":differences,
                "correctness":"not_assessed"}, 1
    before, after = summarize(baseline), summarize(candidate)
    speedup = after["throughput"] / before["throughput"]
    positive(speedup, "computed speedup")
    reduction = 100 * (1 - 1 / speedup)
    if not math.isfinite(reduction):
        raise InvalidInput("Computed reduction overflowed; use a reasonable measurement scale")
    warnings = []
    if baseline["synthetic"] or candidate["synthetic"]:
        warnings.append("Synthetic fixture data: results are not evidence of real training performance.")
    if min(before["run_count"], after["run_count"]) < 3:
        warnings.append("Fewer than three run windows in at least one input; variability is poorly characterized.")
    report = {"status":"comparable", "unit":baseline["measurement"]["unit"],
            "throughput_unit":f'{baseline["measurement"]["unit"]}/second',
            "aggregation":"sum(effective_units) / sum(elapsed_seconds)",
            "baseline":before, "candidate":after, "speedup":speedup,
            "reduction_pct":reduction,
            "reduction_basis":"implied elapsed-time reduction for equal effective work; not total job duration",
            "correctness":"not_assessed", "warnings":warnings}
    if version == 2:
        report["comparison_track"] = "system_optimization"
        report["changed_execution_settings"] = sorted(changed)
        report["contract_check"] = "declared_measurement_and_invariants_match"
        report["provenance"] = {"baseline":baseline["provenance"], "candidate":candidate["provenance"]}
        warnings.append("Contract fields are declarations, not verification of code behavior, gradients, convergence, or task quality.")
        if changed & {"microbatch_size", "gradient_accumulation_steps"}:
            warnings.append("Equal global batch does not prove microbatch equivalence: audit batch-dependent layers, loss normalization, randomness, optimizer cadence, and gradients.")
    return report, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output", help="Write JSON report; cannot overwrite either input")
    args = parser.parse_args()
    try:
        protect_output(args.output, (args.baseline, args.candidate))
        result, code = compare(load_json(args.baseline), load_json(args.candidate))
        emit(result, args.output, (args.baseline, args.candidate))
        return code
    except (InvalidInput, OSError, UnicodeError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
