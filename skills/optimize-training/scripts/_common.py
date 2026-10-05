"""Shared strict JSON input, safe output, and command-line error handling."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import tempfile

class InvalidInput(ValueError):
    """Malformed or insufficient input, reported with exit status 2."""


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidInput(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text):
    def reject(value):
        raise InvalidInput(f"Non-finite JSON number: {value}")
    try:
        def finite_float(value):
            parsed = float(value)
            if not math.isfinite(parsed):
                raise InvalidInput(f"Non-finite JSON number: {value}")
            return parsed
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=reject, parse_float=finite_float)
    except InvalidInput:
        raise
    except (ValueError, UnicodeError) as exc:
        # Python's integer-string digit guard raises plain ValueError rather
        # than JSONDecodeError for excessively long JSON integer literals.
        raise InvalidInput(f"Invalid JSON: {exc}") from exc


def load_json(path):
    return parse_json(Path(path).read_text(encoding="utf-8"))


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInput(f"{label} must be a finite positive number")
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise InvalidInput(f"{label} must be a finite positive number")
    return value


def integer(value, label, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInput(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise InvalidInput(f"{label} must be >= {minimum}")
    return value


def nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise InvalidInput(f"{label} must be a nonempty string")


def protect_output(output, inputs=()):
    if output is None:
        return
    target = Path(output)
    for source in map(Path, inputs):
        if target.resolve() == source.resolve() or (
            target.exists() and source.exists() and os.path.samefile(target, source)
        ):
            raise InvalidInput("--output must not overwrite an input file")


def emit(result, output=None, inputs=()):
    protect_output(output, inputs)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if output:
        target = Path(output)
        # Do not leave truncated output if serialization or writing fails.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".training-report-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(text)
                stream.flush()
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        print(text, end="")
