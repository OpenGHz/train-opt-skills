"""Private deterministic checks. Never copy this file into task workspaces.

This executes participant Python; use only trusted submissions or an isolated host.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def load_pipeline(workdir):
    spec = importlib.util.spec_from_file_location("submission_pipeline", workdir / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(case, workdir):
    checks = []

    def check(name, function):
        try:
            function()
            checks.append({"name": name, "passed": True})
        except Exception as exc:
            checks.append({"name": name, "passed": False, "detail": f"{type(exc).__name__}: {exc}"})

    def require(value, message):
        if not value:
            raise AssertionError(message)

    if case == "async_timing":
        m = load_pipeline(workdir)

        def timed():
            for steps, warmup, cost in [(2, 1, 0.25), (7, 3, 1.5)]:
                state = {"time": 10., "queued": 0, "events": []}
                def step():
                    state["queued"] += cost
                    state["events"].append("step")
                def sync():
                    state["time"] += state["queued"]
                    state["queued"] = 0
                    state["events"].append("sync")
                def clock():
                    state["events"].append("clock")
                    return state["time"]
                elapsed = m.measure(step, sync, clock, steps=steps, warmup=warmup)
                require(math.isclose(elapsed, steps * cost), "elapsed must cover completed measured work only")
                require(state["events"] == ["step"] * warmup + ["sync", "clock"] + ["step"] * steps + ["sync", "clock"], "synchronize window boundaries without serializing every step")
        check("completed_window_and_warmup", timed)

        def invalid():
            for steps, warmup in [(0, 0), (-1, 1), (1, -1)]:
                try:
                    m.measure(lambda: None, lambda: None, lambda: 0., steps=steps, warmup=warmup)
                except ValueError:
                    continue
                raise AssertionError("invalid steps or warmup accepted")
        check("invalid_window_rejected", invalid)
    elif case == "duplicate_shards":
        m = load_pipeline(workdir)

        def coverage():
            for n in [0, 3, 19]:
                for world in [1, 2, 3]:
                    for workers in [0, 1, 4]:
                        actual = []
                        for rank in range(world):
                            for worker in range(max(1, workers)):
                                rows = list(m.shard(list(range(n)), rank, world, worker, workers))
                                expected = list(range(n))[rank * max(1, workers) + worker::world * max(1, workers)]
                                require(rows == expected, "rank × worker stride contract mismatch")
                                actual.extend(rows)
                        require(Counter(actual) == Counter(range(n)), "missing or duplicated samples")
        check("coverage_and_assignment", coverage)

        def invalid():
            for args in [(-1, 2, 0, 1), (2, 2, 0, 1), (0, 0, 0, 1), (0, 1, 1, 0), (0, 1, 0, -1)]:
                try:
                    list(m.shard([1], *args))
                except ValueError:
                    continue
                raise AssertionError(f"invalid shard topology accepted: {args}")
        check("invalid_topologies_rejected", invalid)
    elif case == "augmentation_stats":
        m = load_pipeline(workdir)

        def values():
            for images in [[[0., 2.], [100., 104.]], [[-4., 0., 5.], [8., 9., 10.]], [[4., 4.], [2., 6.]], []]:
                original = [row[:] for row in images]
                actual = m.normalize(images)
                require(len(actual) == len(images), "batch length changed")
                for got, row in zip(actual, images):
                    mean = sum(row) / len(row)
                    var = sum((x - mean) ** 2 for x in row) / len(row)
                    expected = [(x - mean) / math.sqrt(var + 1e-5) for x in row]
                    require(len(got) == len(row) and all(math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8) for a, b in zip(got, expected)), "per-image population-statistics mismatch")
                require(images == original, "input was mutated")
        check("per_image_reference_and_no_mutation", values)

        def independence():
            alone = m.normalize([[1., 3., 9.]])[0]
            together = m.normalize([[1., 3., 9.], [500., 800., 900.]])[0]
            require(alone == together, "one image depends on unrelated batch members")
            try:
                m.normalize([[]])
            except ValueError:
                return
            raise AssertionError("empty image was not rejected")
        check("batch_independence_and_empty_image", independence)
    elif case == "weight_reorder":
        m = load_pipeline(workdir)

        def reorder():
            for heads in [1, 2, 4]:
                src = [[float(row * 3 + col) for col in range(3)] for row in range(heads * 3)]
                original = [row[:] for row in src]
                expected = [src[3 * h + part][:] for part in range(3) for h in range(heads)]
                actual = m.convert_qkv(src, heads)
                require(actual == expected, "same-shape QKV row semantics are wrong")
                require(src == original, "conversion mutated source")
                vector = [0.3, -1.1, 2.]
                projection = lambda matrix: [sum(a * b for a, b in zip(row, vector)) for row in matrix]
                require(projection(actual) == projection(expected), "projection behavior mismatch")
        check("qkv_layout_and_projection", reorder)

        def invalid():
            for matrix, heads in [([[1.]] * 5, 2), ([[1.], [1., 2.], [3.]], 1), ([[1.]], 0)]:
                try:
                    m.convert_qkv(matrix, heads)
                except ValueError:
                    continue
                raise AssertionError("malformed QKV input accepted")
        check("malformed_layout_rejected", invalid)
    elif case == "protocol_change":
        obj = json.loads((workdir / "assessment.json").read_text())
        check("observed_ratio", lambda: require(math.isclose(obj["observed_step_rate_ratio"], 2.0), "observed ratio should use steps / elapsed seconds"))
        check("equivalence_rejected", lambda: require(obj["equivalent_training_speedup_established"] is False and obj["task_quality_verified"] is False, "changed workload and short loss cannot establish equivalence or task quality"))
        check("changed_inputs_identified", lambda: require(set(obj["changed_fields"]) == {"real_cameras", "image_resolution"}, "changed input protocol fields must be explicit"))
        check("unmeasured_claims_absent", lambda: require(obj["time_to_target_speedup"] is None, "unmeasured target quality cannot have numeric speedup"))
    elif case == "optional_integration":
        obj = json.loads((workdir / "integration-result.json").read_text())
        audit = json.loads((workdir / "audit.json").read_text())
        check("integration_absence_recorded", lambda: require(obj["integration_status"] == "unavailable" and obj["external_skill_executed"] is False, "missing optional skill cannot be recorded as executed"))
        check("fallback_audit_completed", lambda: require(audit["records"] == 4 and audit["unique_ids"] == 2 and set(audit["missing_ids"]) == {"c", "d"} and set(audit["repeated_ids"]) == {"a", "b"}, "actual local audit mismatch"))
        check("audit_failure_propagated", lambda: require(obj["data_contract_passed"] is False and obj["gpu_performance_measured"] is False, "duplicate coverage and unavailable GPU must remain unvalidated"))
        def rerun_audit():
            with tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                samples = folder / "samples.jsonl"
                expected = folder / "expected.json"
                output = folder / "audit.json"
                samples.write_text(''.join(json.dumps({"sample_id": value}) + '\n' for value in ["x", "y", "y", "z", "z"]))
                expected.write_text(json.dumps(["w", "x", "y", "z"]))
                completed = subprocess.run([sys.executable, str(workdir / "audit.py"), "--samples", str(samples), "--expected", str(expected), "--output", str(output)], capture_output=True, text=True, timeout=5)
                # Either zero (audit ran) or one (detected a mismatch) is valid.
                require(completed.returncode in [0, 1], "audit script did not run")
                got = json.loads(output.read_text())
                require(got["records"] == 5 and got["unique_ids"] == 3 and set(got["missing_ids"]) == {"w"} and set(got["repeated_ids"]) == {"y", "z"}, "audit script does not generalize to unseen IDs")
        check("reproducible_audit_on_new_inputs", rerun_audit)
    else:
        raise ValueError(f"unknown case: {case}")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate(args.case, args.workdir.resolve())
    except Exception as exc:
        result = [{"name": "artifact_load", "passed": False, "detail": f"{type(exc).__name__}: {exc}"}]
    print(json.dumps(result, ensure_ascii=False))
