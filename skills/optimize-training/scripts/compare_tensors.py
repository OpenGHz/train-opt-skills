#!/usr/bin/env python3
"""Compare numeric NPZ snapshots without pickle; requires NumPy.

Positional inputs: reference.npz candidate.npz. Exact keys and shapes required.
Only real integer/floating dtypes are accepted; object, string, complex and bool
arrays are rejected. Non-finite values always fail. --atol and --rtol are explicit,
finite and nonnegative. Element acceptance: abs(candidate-reference) <= atol +
rtol*abs(reference). This is asymmetric: reference supplies the relative scale.
Report max relative error uses abs(reference); zero reference and nonzero error
is unbounded (null numeric value plus unbounded=true). Processing uses bounded
65536-element arithmetic chunks and emits one summary per key, no tensor values.
NPZ archives must contain only .npy members. Members must still decompress into
RAM; use trusted, reasonably sized files. On platforms with a narrow longdouble,
integer magnitudes beyond its consecutive-integer range and wider float dtypes
are rejected explicitly to prevent precision-loss false matches.
Exit: 0 match; 1 key/shape/value/non-finite mismatch; 2 malformed/unsupported input.
"""
from __future__ import annotations
import argparse
import math
import sys
import zipfile
from _common import InvalidInput, emit, protect_output


def tolerance(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Tolerance must be finite and nonnegative") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Tolerance must be finite and nonnegative")
    return number


def compare(reference_path, candidate_path, atol, rtol):
    try:
        import numpy as np
    except ImportError as exc:
        raise InvalidInput("NumPy is required: install numpy in the environment used to run this script") from exc
    try:
        # np.load accepts arbitrary ZIP members and returns their raw bytes.
        # Reject these even when a member occurs in only one input archive.
        for path in (reference_path, candidate_path):
            with zipfile.ZipFile(path) as archive:
                if any(not item.filename.endswith(".npy") for item in archive.infolist()):
                    raise InvalidInput("NPZ archives may contain only .npy tensor members")
        reference = np.load(reference_path, allow_pickle=False)
        candidate = np.load(candidate_path, allow_pickle=False)
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as exc:
        raise InvalidInput(f"Cannot open numeric NPZ inputs: {exc}") from exc
    if not isinstance(reference, np.lib.npyio.NpzFile) or not isinstance(candidate, np.lib.npyio.NpzFile):
        for value in (reference, candidate):
            if hasattr(value, "close"):
                value.close()
        raise InvalidInput("Both inputs must be .npz archives, not .npy arrays")
    with reference, candidate:
        if len(reference.files) != len(set(reference.files)) or len(candidate.files) != len(set(candidate.files)):
            raise InvalidInput("Duplicate NPZ member keys are not allowed")
        ref_keys, cand_keys = set(reference.files), set(candidate.files)
        if not ref_keys or not cand_keys:
            raise InvalidInput("NPZ archives must contain at least one tensor")
        missing, extra = sorted(ref_keys - cand_keys), sorted(cand_keys - ref_keys)
        failed = bool(missing or extra)
        results = []
        for key in sorted(ref_keys & cand_keys):
            try:
                ref, cand = reference[key], candidate[key]
            except (ValueError, TypeError, EOFError, zipfile.BadZipFile) as exc:
                raise InvalidInput(f"Cannot load numeric tensor {key}: {exc}") from exc
            if not isinstance(ref, np.ndarray) or not isinstance(cand, np.ndarray):
                raise InvalidInput(f"{key}: each NPZ member must decode to a NumPy array")
            if ref.dtype.kind not in "iuf" or cand.dtype.kind not in "iuf":
                raise InvalidInput(f"{key}: only real integer/floating arrays are supported")
            # NumPy longdouble is only float64 on some platforms. Casting an
            # unsafe integer (including in mixed int/float comparisons) can
            # erase a one-unit difference before the tolerance check. Reject
            # conservatively rather than claim these snapshots match.
            working_info = np.finfo(np.longdouble)
            integer_limit = 1 << (working_info.nmant + 1)
            for array in (ref, cand):
                if array.dtype.kind in "iu" and array.size:
                    minimum, maximum = int(array.min()), int(array.max())
                    if minimum < -integer_limit or maximum > integer_limit:
                        raise InvalidInput(
                            f"{key}: integer magnitude exceeds the exact consecutive-integer "
                            f"range of this platform's comparison dtype (2**{working_info.nmant + 1}); "
                            "comparison refused to avoid precision loss"
                        )
                elif array.dtype.kind == "f":
                    source_info = np.finfo(array.dtype)
                    if source_info.nmant > working_info.nmant or source_info.maxexp > working_info.maxexp:
                        raise InvalidInput(f"{key}: source float dtype exceeds this platform's comparison precision/range")
            item = {"key":key, "reference_shape":list(ref.shape), "candidate_shape":list(cand.shape),
                    "reference_dtype":str(ref.dtype), "candidate_dtype":str(cand.dtype)}
            if ref.shape != cand.shape:
                item["status"] = "shape_mismatch"
                failed = True
                results.append(item)
                continue
            mismatches = nonfinite = 0
            max_abs = max_rel = np.longdouble(0)
            unbounded = False
            # Precision guards above make the working cast safe for supported inputs.
            iterator = np.nditer([ref,cand], flags=["external_loop","buffered","zerosize_ok"],
                                 op_flags=[["readonly"],["readonly"]], buffersize=65536)
            for left,right in iterator:
                left, right = left.astype(np.longdouble), right.astype(np.longdouble)
                finite = np.isfinite(left) & np.isfinite(right)
                nonfinite += int((~finite).sum())
                if not finite.any():
                    continue
                left, right = left[finite], right[finite]
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    error = np.abs(right-left)
                    bound = np.longdouble(atol) + np.longdouble(rtol)*np.abs(left)
                    # Infinite intermediate errors are never silently accepted.
                    mismatches += int(((error > bound) | ~np.isfinite(error)).sum())
                    max_abs = max(max_abs, error.max(initial=0))
                    zeros = left == 0
                    unbounded |= bool(((error > 0) & zeros).any())
                    rel = error[~zeros] / np.abs(left[~zeros])
                    max_rel = max(max_rel, rel.max(initial=0))
            def finite_float(value):
                return float(value) if np.isfinite(value) and value <= sys.float_info.max else None
            item.update(status="match" if not (mismatches or nonfinite) else "value_mismatch",
                        elements=int(ref.size), mismatched_finite_elements=mismatches,
                        nonfinite_pairs=nonfinite, max_abs_error=finite_float(max_abs),
                        max_abs_error_exceeds_float_range=finite_float(max_abs) is None,
                        max_relative_error=None if unbounded else finite_float(max_rel),
                        relative_error_unbounded=unbounded,
                        max_relative_error_exceeds_float_range=(not unbounded and finite_float(max_rel) is None))
            failed |= bool(mismatches or nonfinite)
            results.append(item)
        return {"status":"mismatch" if failed else "match", "atol":atol, "rtol":rtol,
                "relative_scale":"reference", "missing_keys":missing, "extra_keys":extra,
                "tensors":results, "scope":"snapshot comparison only; not full training correctness"}, 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", required=True, type=tolerance)
    parser.add_argument("--rtol", required=True, type=tolerance)
    parser.add_argument("--output", help="Write JSON report; cannot overwrite an input")
    args = parser.parse_args()
    try:
        protect_output(args.output, (args.reference, args.candidate))
        result, code = compare(args.reference, args.candidate, args.atol, args.rtol)
        emit(result, args.output, (args.reference, args.candidate))
        return code
    except (InvalidInput, OSError, ValueError, TypeError, OverflowError, EOFError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
