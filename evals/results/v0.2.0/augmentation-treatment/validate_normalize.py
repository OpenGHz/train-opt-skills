"""Small CPU-only correctness checks; no training or timing claims."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys

from pipeline import normalize


# Python binary64 fixtures: small arithmetic-rounding differences against the
# independently accumulated statistics reference are allowed. Batch isolation
# and shape/input preservation are checked exactly.
ATOL = RTOL = 1e-12
ROOT = Path(__file__).resolve().parent


def reference(image):
    mean = statistics.mean(image)
    variance = statistics.pvariance(image)
    return [(value - mean) / math.sqrt(variance + 1e-5) for value in image]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "candidate"), required=True)
    args = parser.parse_args()
    rng = random.Random(401)
    fixtures = [
        [[1, 3], [100, 200, 300]],
        [[-7, -7, -7], [42]],
        [[-3.5, 0.0, 8.25], [1e-8, 2e-8, -1e-8]],
        [[0, 255, 0, 255], [0.125, 0.25]],
    ]
    fixtures.extend(
        [[rng.uniform(-1000, 1000) for _ in range(rng.randrange(1, 13))]
         for _ in range(rng.randrange(1, 6))]
        for _ in range(24)
    )
    records = []
    metrics = {}

    def run(name, check):
        try:
            check()
        except Exception as exc:
            records.append({"name": name, "status": "failed", "error": repr(exc)})
        else:
            records.append({"name": name, "status": "passed"})

    def reference_parity():
        maximum = 0.0
        compared = 0
        for batch in fixtures:
            actual = normalize(batch)
            assert len(actual) == len(batch)
            for image, output in zip(batch, actual):
                expected = reference(image)
                assert len(output) == len(expected)
                for got, want in zip(output, expected):
                    assert math.isfinite(got)
                    maximum = max(maximum, abs(got - want))
                    compared += 1
                    assert math.isclose(got, want, abs_tol=ATOL, rel_tol=RTOL), (got, want)
        metrics.update(reference_max_abs_error=maximum, pixels_compared=compared)

    def population_and_epsilon():
        expected = [-1 / math.sqrt(1 + 1e-5), 1 / math.sqrt(1 + 1e-5)]
        assert normalize([[1, 3]])[0] == expected
        expected_tiny = [-1e-8 / math.sqrt(1e-16 + 1e-5),
                         1e-8 / math.sqrt(1e-16 + 1e-5)]
        assert normalize([[-1e-8, 1e-8]])[0] == expected_tiny

    def batch_isolation():
        for batch in fixtures:
            existing = normalize(batch)
            unrelated = [1000000, 2000000, -3000000]
            assert normalize(batch + [unrelated])[:-1] == existing
            assert normalize([unrelated] + batch)[1:] == existing
            assert normalize(list(reversed(batch))) == list(reversed(existing))
            for index, image in enumerate(batch):
                assert existing[index] == normalize([image])[0]
                replaced = copy.deepcopy(batch)
                for other in range(len(batch)):
                    if other != index:
                        replaced[other] = unrelated
                assert normalize(replaced)[index] == existing[index]

    def shape_and_input():
        for batch in fixtures:
            saved = copy.deepcopy(batch)
            result = normalize(batch)
            assert batch == saved
            assert isinstance(result, list) and result is not batch
            assert [len(row) for row in result] == [len(row) for row in batch]
            assert all(isinstance(row, list) and row is not original
                       for row, original in zip(result, batch))
            result[0][0] = 123456
            assert batch == saved

    def empty_batch():
        assert normalize([]) == []

    def empty_images():
        for batch in ([[]], [[], [1, 2]], [[1, 2], []]):
            saved = copy.deepcopy(batch)
            try:
                normalize(batch)
            except ValueError:
                pass
            else:
                raise AssertionError("empty image did not raise ValueError")
            assert batch == saved

    def constant_images():
        assert normalize([[2, 2, 2], [-999], [0, 0]]) == [[0, 0, 0], [0], [0, 0]]

    for name, check in (
        ("independent_reference_parity", reference_parity),
        ("population_variance_and_epsilon", population_and_epsilon),
        ("batch_neighbor_and_order_invariance", batch_isolation),
        ("ragged_list_shape_and_input_immutability", shape_and_input),
        ("empty_batch", empty_batch),
        ("empty_image_errors", empty_images),
        ("constant_and_singleton_images", constant_images),
    ):
        run(name, check)

    passed = all(record["status"] == "passed" for record in records)
    report = {
        "phase": args.phase,
        "status": "passed" if passed else "failed",
        "command": f"python validate_normalize.py --phase {args.phase}",
        "exit_status": 0 if passed else 1,
        "device": "CPU",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pipeline_sha256": hashlib.sha256((ROOT / "pipeline.py").read_bytes()).hexdigest(),
        "reference": "statistics.mean and statistics.pvariance, epsilon=1e-5",
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "fixture_batches": fixtures,
        "checks": records,
        "metrics": metrics,
        "scope": "Deterministic statistics only; no stochastic augmentation, GPU, performance, or training validation.",
    }
    destination = ROOT / f"{args.phase}_validation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"artifact": destination.name, "status": report["status"],
                      "checks": records, "metrics": metrics}, indent=2))
    return report["exit_status"]


if __name__ == "__main__":
    sys.exit(main())
