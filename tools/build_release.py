#!/usr/bin/env python3
"""Build reproducible project and standalone-skill ZIP archives with checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

SKILL_NAME = "optimize-training"
EXCLUDED_DIRS = {".git", "__pycache__", "dist", ".venv", "venv", "node_modules"}
MANIFEST_NAME = "RELEASE-MANIFEST.json"


def collect_files(source: Path, output_dir: Path) -> list[tuple[str, bytes]]:
    result: list[tuple[str, bytes]] = []
    for directory, names, filenames in os.walk(source, followlinks=False):
        parent = Path(directory)
        retained = []
        for name in sorted(names):
            path = parent / name
            if name in EXCLUDED_DIRS or path.resolve() == output_dir:
                continue
            if path.is_symlink():
                raise ValueError(f"refusing to package symlink: {path}")
            retained.append(name)
        names[:] = retained
        for name in sorted(filenames):
            path = parent / name
            if path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"refusing to package non-regular file: {path}")
            relative = path.relative_to(source).as_posix()
            if relative == MANIFEST_NAME:
                raise ValueError(f"reserved generated filename in source: {MANIFEST_NAME}")
            result.append((relative, path.read_bytes()))
    return sorted(result)


def _write_archive(path: Path, prefix: str, version: str, files: list[tuple[str, bytes]]) -> None:
    manifest = {
        "format_version": 1,
        "project_version": version,
        "files": [
            {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in files
        ],
    }
    entries = files + [(MANIFEST_NAME, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries):
            info = zipfile.ZipInfo(f"{prefix}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_archives(root: Path, output_dir: Path | None = None) -> list[Path]:
    root = root.resolve()
    output_dir = (output_dir or root / "dist").resolve()
    skill = root / "skills" / SKILL_NAME
    if output_dir == root or output_dir.is_relative_to(skill):
        raise ValueError("output directory must not be the project root or inside the skill")
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?", version):
        raise ValueError("VERSION must be a semantic version, e.g. 0.1.0")
    if skill.is_symlink() or not (skill / "SKILL.md").is_file():
        raise ValueError(f"missing regular skill directory: {skill}")
    project_files = collect_files(root, output_dir)
    skill_files = collect_files(skill, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_archive = output_dir / f"train-opt-skills-{version}.zip"
    skill_archive = output_dir / f"{SKILL_NAME}-{version}.zip"
    _write_archive(project_archive, "train-opt-skills", version, project_files)
    _write_archive(skill_archive, SKILL_NAME, version, skill_files)
    paths = [project_archive, skill_archive]
    checksums = "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in paths)
    (output_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, help="Default: <root>/dist")
    args = parser.parse_args(argv)
    try:
        paths = build_archives(args.root, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    print(paths[0].parent / "SHA256SUMS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
