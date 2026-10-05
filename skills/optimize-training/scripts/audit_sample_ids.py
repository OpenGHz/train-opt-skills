#!/usr/bin/env python3
"""Audit observed sample IDs across ranks/workers, separately within each epoch.

Input: JSONL objects {epoch:nonnegative-int,rank:nonnegative-int,
worker:nonnegative-int,sample_id:string-or-int}. Use worker=0 for num_workers=0.
Optional --expected: JSON array of unique string-or-int IDs expected in EVERY
observed epoch. ID type is significant: 1 and "1" differ. Empty strings, booleans,
null, containers, blank files and missing fields are rejected. Blank lines ignored.
--allow-repeats makes duplicates diagnostic (e.g. expected sampler padding), but
does not excuse missing/extra samples. Without --expected, coverage is unknown.
The audit cannot detect completely unobserved epochs or workers. Worker count and
ordering are not inferred. Output examples are capped, counts are complete.
Exit: 0 no detected mismatch; 1 duplicates/missing/extra; 2 malformed input.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from _common import InvalidInput, emit, integer, load_json, parse_json, protect_output


def identity(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
        raise InvalidInput("sample IDs must be nonempty strings or integers (not booleans)")
    return type(value).__name__, value


def audit(path, expected=None, allow_repeats=False):
    expected_ids = None
    if expected is not None:
        if not isinstance(expected, list):
            raise InvalidInput("Expected IDs must be a JSON array")
        expected_ids = set(identity(value) for value in expected)
        if len(expected_ids) != len(expected):
            raise InvalidInput("Expected IDs must not contain duplicates")
    epochs = defaultdict(Counter)
    owners = defaultdict(set)
    records = 0
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = parse_json(line)
            if not isinstance(row, dict):
                raise InvalidInput(f"Line {number}: expected an object")
            for field in ("epoch", "rank", "worker"):
                integer(row.get(field), f"Line {number}: {field}", 0)
            key = identity(row.get("sample_id"))
            epochs[row["epoch"]][key] += 1
            owners[(row["epoch"], key)].add((row["rank"], row["worker"]))
            records += 1
    if not records:
        raise InvalidInput("No sample records found")
    reports = []
    mismatch = False
    sort_key = lambda value: (value[0], str(value[1]))
    for epoch, counts in sorted(epochs.items()):
        repeated = [key for key, count in counts.items() if count > 1]
        missing = set() if expected_ids is None else expected_ids - counts.keys()
        extra = set() if expected_ids is None else counts.keys() - expected_ids
        mismatch |= bool((repeated and not allow_repeats) or missing or extra)
        reports.append({"epoch":epoch, "records":sum(counts.values()), "unique_ids":len(counts),
                        "repeated_ids":len(repeated), "repeat_occurrences":sum(counts[key]-1 for key in repeated),
                        "cross_owner_repeated_ids":sum(len(owners[(epoch,key)]) > 1 for key in repeated),
                        "repeat_examples":[{"sample_id":key[1], "count":counts[key],
                                            "owners":[{"rank":rank,"worker":worker} for rank,worker in sorted(owners[(epoch,key)])[:20]]}
                                           for key in sorted(repeated,key=sort_key)[:20]],
                        "missing_count":None if expected_ids is None else len(missing),
                        "extra_count":None if expected_ids is None else len(extra),
                        "missing_examples":[key[1] for key in sorted(missing,key=sort_key)[:20]],
                        "extra_examples":[key[1] for key in sorted(extra,key=sort_key)[:20]]})
    return {"status":"mismatch" if mismatch else "no_detected_mismatch",
            "coverage":"unknown" if expected_ids is None else "checked_for_observed_epochs",
            "allow_repeats":allow_repeats, "records":records, "epochs":reports,
            "limitations":["Cannot detect entirely unobserved epochs or workers.",
                           "Observed ID coverage does not establish augmentation, ordering, or training correctness."]}, 1 if mismatch else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", help="JSONL sample records")
    parser.add_argument("--expected", help="JSON array of expected IDs in each observed epoch")
    parser.add_argument("--allow-repeats", action="store_true")
    parser.add_argument("--output", help="Write JSON report; cannot overwrite an input")
    args = parser.parse_args()
    try:
        inputs = [args.samples] + ([args.expected] if args.expected else [])
        protect_output(args.output, inputs)
        result, code = audit(args.samples, load_json(args.expected) if args.expected else None, args.allow_repeats)
        emit(result, args.output, inputs)
        return code
    except (InvalidInput, OSError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
