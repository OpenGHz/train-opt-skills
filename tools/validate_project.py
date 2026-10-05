#!/usr/bin/env python3
"""Validate the distributable project's metadata, files, JSON, and local links."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

SKILL_NAME = "optimize-training"
REQUIRED_FILES = (
    "README.md", "README.zh-CN.md", "LICENSE", "NOTICE", "CHANGELOG.md",
    "CONTRIBUTING.md", "AGENTS.md", "VERSION", "plugin.json",
    "docs/installation.md", "docs/tool-reference.md", "docs/validation.md",
    "docs/design-references.md", "docs/training_performance_optimization_guide_zh_v2.md",
    f"skills/{SKILL_NAME}/SKILL.md",
    f"skills/{SKILL_NAME}/assets/experiment.json",
    f"skills/{SKILL_NAME}/assets/report-template.md",
)
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", "dist", "node_modules"}


def _source_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
            if path.is_file() and not path.is_symlink():
                yield path


def _scalar(value: str) -> str:
    if value.startswith('"'):
        result = json.loads(value)
        if not isinstance(result, str):
            raise ValueError("expected a string")
        return result
    if value.startswith("'"):
        if not value.endswith("'") or len(value) < 2:
            raise ValueError("unclosed single-quoted string")
        return value[1:-1].replace("''", "'")
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def frontmatter_fields(text: str) -> dict[str, str]:
    """Read string name/description fields without imposing a YAML dependency."""
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("frontmatter has no closing ---") from exc
    result: dict[str, str] = {}
    seen: set[str] = set()
    index = 1
    while index < end:
        match = re.match(r"^([A-Za-z][A-Za-z0-9_-]*):\s*(.*)$", lines[index])
        if not match:
            index += 1
            continue
        key, value = match.groups()
        if key in seen:
            raise ValueError(f"duplicate frontmatter key: {key}")
        seen.add(key)
        if key in {"name", "description"}:
            if value in {">", ">-", ">+", "|", "|-", "|+"}:
                block = []
                index += 1
                while index < end and (not lines[index] or lines[index][0].isspace()):
                    block.append(lines[index].strip())
                    index += 1
                result[key] = (" " if value.startswith(">") else "\n").join(block).strip()
                continue
            result[key] = _scalar(value)
        index += 1
    return result


def _without_fences(text: str) -> str:
    output = []
    fence: str | None = None
    fence_length = 0
    for line in text.splitlines():
        found = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence:
            if found and found[1][0] == fence and len(found[1]) >= fence_length:
                fence = None
            continue
        if found:
            fence, fence_length = found[1][0], len(found[1])
            continue
        output.append(line)
    return "\n".join(output)


def _destination(text: str) -> str:
    text = text.strip()
    if text.startswith("<"):
        end = text.find(">")
        return text[1:end] if end >= 0 else text
    # Markdown titles are separated from the destination by whitespace.
    return re.split(r"(?<!\\)\s+", text, maxsplit=1)[0]


def markdown_destinations(text: str):
    text = _without_fences(text)
    for match in re.finditer(r"!?\[[^\]\n]*\]\(", text):
        start = match.end()
        index, depth = start, 1
        while index < len(text) and depth:
            if text[index] == "\\":
                index += 2
                continue
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
            index += 1
        if depth == 0:
            yield _destination(text[start:index - 1])
    for match in re.finditer(r"(?m)^\s{0,3}\[[^\]\n]+\]:\s*(.+)$", text):
        yield _destination(match[1])


def validate_project(root: Path) -> list[str]:
    root = root.resolve()
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing required regular file: {relative}")
    skill = root / "skills" / SKILL_NAME
    for folder, pattern in (("scripts", "*.py"), ("references", "*.md")):
        if not any((skill / folder).glob(pattern)):
            errors.append(f"skill needs at least one {folder}/{pattern} file")
    entry = skill / "SKILL.md"
    if entry.is_file():
        try:
            text = entry.read_text(encoding="utf-8")
            fields = frontmatter_fields(text)
            name = fields.get("name", "")
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 64:
                errors.append("SKILL.md name must be lowercase kebab-case, 1–64 characters")
            if name != SKILL_NAME:
                errors.append(f"SKILL.md name must match directory: {SKILL_NAME}")
            description = fields.get("description", "")
            if not description.strip() or len(description) > 1024:
                errors.append("SKILL.md description must contain 1–1024 characters")
            if len(text.splitlines()) > 500:
                errors.append("SKILL.md exceeds the 500-line limit")
        except (ValueError, UnicodeError) as exc:
            errors.append(f"SKILL.md frontmatter: {exc}")
    for path in _source_files(root):
        relative = path.relative_to(root)
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as exc:
                errors.append(f"{relative}: invalid JSON: {exc}")
        elif path.suffix.lower() == ".md":
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError as exc:
                errors.append(f"{relative}: invalid UTF-8: {exc}")
                continue
            for destination in markdown_destinations(text):
                parsed = urlsplit(destination)
                if not destination or parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                local_path = re.sub(r"\\([\\ ()])", r"\1", unquote(parsed.path))
                resolved = (path.parent / local_path).resolve()
                if not resolved.is_relative_to(root):
                    errors.append(f"{relative}: local link escapes project: {destination}")
                elif not resolved.exists():
                    errors.append(f"{relative}: broken local link: {destination}")
    version_file = root / "VERSION"
    if version_file.is_file():
        version = version_file.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?", version):
            errors.append("VERSION must be a semantic version, e.g. 0.1.0")
        manifest = root / "plugin.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or data.get("version") != version:
                    errors.append("plugin.json version must match VERSION")
            except ValueError:
                pass  # Already reported by the JSON check.
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    errors = validate_project(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Project validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
