#!/usr/bin/env python3
"""Copy the bundled skill to an explicitly chosen skill-directory parent."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

SKILL_NAME = "optimize-training"


def install_skill(source: Path, target: Path, *, dry_run: bool = False) -> Path:
    """Never overwrite a destination; reject symlinks within the source tree."""
    if source.is_symlink():
        raise ValueError("source skill directory must not be a symlink")
    source = source.resolve()
    if not source.is_dir() or not (source / "SKILL.md").is_file():
        raise ValueError(f"source is not a skill directory: {source}")
    for directory, names, filenames in os.walk(source, followlinks=False):
        for name in names + filenames:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"source contains a symlink: {path.relative_to(source)}")
            if not path.is_dir() and not path.is_file():
                raise ValueError(f"source contains a non-regular file: {path.relative_to(source)}")
    target = target.expanduser().resolve()
    destination = target / SKILL_NAME
    if target.is_relative_to(source):
        raise ValueError("target must not be inside the source skill directory")
    if target.exists() and not target.is_dir():
        raise ValueError(f"target parent is not a directory: {target}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing destination: {destination}")
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
        # copytree's default dirs_exist_ok=False rejects races creating destination.
        shutil.copytree(source, destination, symlinks=False)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True,
                        help="Parent directory of installed skills, e.g. /project/.agents/skills")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print destination without writing")
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parents[1] / "skills" / SKILL_NAME
    try:
        destination = install_skill(source, args.target, dry_run=args.dry_run)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{'Would install' if args.dry_run else 'Installed'}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
