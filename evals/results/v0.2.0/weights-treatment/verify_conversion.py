"""Bounded CPU-only regression checks for Q/K/V conversion."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from pipeline import convert_qkv


def check_projection(weight, num_heads, inputs):
    before = copy.deepcopy(weight)
    converted = convert_qkv(weight, num_heads)
    assert weight == before, "Conversion mutated its input"
    assert converted is not weight
    assert all(result_row is not source_row for result_row in converted for source_row in weight)
    assert len(converted) == len(weight)
    assert all(len(row) == len(weight[0]) for row in converted)

    x = np.asarray(inputs, dtype=np.float64)
    actual = x @ np.asarray(converted, dtype=np.float64).T
    # Calculate Q, K and V separately using the original per-head projections.
    q, k, v = [], [], []
    for head in range(num_heads):
        q.append(x @ np.asarray(weight[3 * head], dtype=np.float64))
        k.append(x @ np.asarray(weight[3 * head + 1], dtype=np.float64))
        v.append(x @ np.asarray(weight[3 * head + 2], dtype=np.float64))
    reference = np.column_stack(q + k + v)
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    # Small integer/dyadic inputs have exactly representable sums in float64.
    np.testing.assert_allclose(actual, reference, atol=0.0, rtol=0.0)
    converted[0][0] = "modified output"
    assert weight == before, "Returned rows alias the input"
    return float(np.max(np.abs(actual - reference)))


def main():
    fixture = json.loads(Path("projection_fixture.json").read_text())
    errors = [check_projection(fixture["weight"], fixture["num_heads"], fixture["inputs"])]
    for heads in (1, 2, 3, 5):
        for width in (1, 4, 7):
            weight = [
                [(row + 1) * (column + 2) - column * column for column in range(width)]
                for row in range(3 * heads)
            ]
            probes = np.vstack((np.eye(width), np.arange(width) - 1.5))
            errors.append(check_projection(weight, heads, probes))

    valid = [[1, 2], [3, 4], [5, 6]]
    invalid = [
        (valid, 0), (valid, -1), (valid, 1.5), (valid, True),
        (valid, "1"), (valid, None), ([], 1), (valid[:2], 1),
        (valid + [[7, 8]], 1), (valid, 2), ([[], [], []], 1),
        ([[1], [2, 3], [4]], 1), ([[1], None, [3]], 1),
        ([[1], 2, [3]], 1), ([[1], "x", [3]], 1),
        ([1, 2, 3], 1), (None, 1), (tuple(valid), 1),
        ([(1, 2), [3, 4], [5, 6]], 1),
    ]
    for weight, heads in invalid:
        before = copy.deepcopy(weight)
        try:
            convert_qkv(weight, heads)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid layout was accepted: {weight!r}, {heads!r}")
        assert weight == before
    assert convert_qkv(valid, np.int64(1)) == valid

    result = {
        "status": "passed",
        "command": "python verify_conversion.py",
        "device": "CPU",
        "dtype": "float64",
        "atol": 0.0,
        "rtol": 0.0,
        "projection_cases": len(errors),
        "invalid_layout_cases": len(invalid),
        "max_absolute_error": max(errors),
        "all_values_finite": True,
        "input_preserved_and_output_rows_independent": True,
        "converter_sha256": hashlib.sha256(Path("pipeline.py").read_bytes()).hexdigest(),
        "fixture": "projection_fixture.json",
    }
    Path("verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
