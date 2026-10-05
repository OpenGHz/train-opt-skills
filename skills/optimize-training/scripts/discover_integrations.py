#!/usr/bin/env python3
"""Inventory explicitly supplied local skills without importing or running them."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from _common import emit, load_json

DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "assets" / "integrations.json"
NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _scalar(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def frontmatter(text):
    """Read only literal name and metadata.version; never evaluate YAML tags."""
    lines = text.lstrip("\ufeff").splitlines()
    if not lines or lines[0].strip() != "---":
        return None, None
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return None, None
    names = []
    version = None
    in_metadata = False
    for line in lines[1:end]:
        if line.startswith("name:"):
            names.append(_scalar(line.split(":", 1)[1]))
        if line and not line[0].isspace():
            in_metadata = line.strip() == "metadata:"
        elif in_metadata and re.match(r"^\s+version:\s*", line):
            value = _scalar(line.split(":", 1)[1])
            if value and value not in ("|", ">", "null", "~"):
                version = value
    if len(names) != 1 or NAME.fullmatch(names[0]) is None:
        return None, version
    return names[0], version


def _has_symlink(path):
    return any(part.is_symlink() for part in (path, *path.parents))


def _safe_under(path, root):
    if _has_symlink(path):
        return False
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def _candidate_files(root, warnings):
    """Scan root, its direct children, and root/skills direct children only."""
    if _has_symlink(root):
        warnings.append({"path": str(root), "reason": "root_or_ancestor_is_symlink"})
        return []
    if not root.is_dir():
        warnings.append({"path": str(root), "reason": "root_not_directory"})
        return []
    candidates = [root / "SKILL.md"]
    bases = [root]
    nested = root / "skills"
    if nested.is_dir() and _safe_under(nested, root):
        bases.append(nested)
    for base in bases:
        try:
            children = sorted(base.iterdir())
        except OSError as exc:
            warnings.append({"path": str(base), "reason": "unreadable_directory", "detail": str(exc)})
            continue
        for child in children:
            if child.is_symlink():
                warnings.append({"path": str(child), "reason": "symlink_not_followed"})
            elif child.is_dir():
                candidates.append(child / "SKILL.md")
    result = []
    for path in candidates:
        if path.is_symlink():
            warnings.append({"path": str(path), "reason": "symlink_not_followed"})
        elif path.is_file() and _safe_under(path, root):
            result.append(path.resolve())
    return result


def git_metadata(directory):
    """Read repository identity locally; do not inspect remotes, fetch, or run hooks."""
    executable = shutil.which("git")
    if not executable:
        return {"status": "unavailable", "reason": "git_not_found", "commit": None}
    env = {"PATH": os.environ.get("PATH", ""), "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        values = []
        for argument in ("--show-toplevel", "HEAD"):
            completed = subprocess.run(
                [executable, "-C", str(directory), "rev-parse", argument],
                capture_output=True, text=True, check=True, timeout=2, env=env,
            )
            values.append(completed.stdout.strip())
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", values[1]) is None:
            raise ValueError("unexpected commit identifier")
        return {"status": "available", "repository_root": values[0], "commit": values[1],
                "scope": "containing_repository_only; origin_and_worktree_cleanliness_not_verified"}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"status": "unavailable", "reason": "not_a_readable_git_repository", "commit": None}


def package_inventory(dependencies):
    result = []
    for dependency in dependencies:
        entry = dict(dependency)
        try:
            entry.update(status="metadata_present", version=importlib.metadata.version(dependency["distribution"]))
        except importlib.metadata.PackageNotFoundError:
            entry.update(status="metadata_absent", version=None)
        except (OSError, ValueError):
            entry.update(status="metadata_unavailable", version=None)
        entry["runtime_import_tested"] = False
        result.append(entry)
    return result


def validate_registry(registry):
    if not isinstance(registry, dict) or type(registry.get("schema_version")) is not int or registry["schema_version"] != 1:
        raise ValueError("registry schema_version must be 1")
    specs = registry.get("integrations")
    if not isinstance(specs, list) or not specs:
        raise ValueError("registry integrations must be a nonempty list")
    seen = set()
    for spec in specs:
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str) or NAME.fullmatch(spec["name"]) is None:
            raise ValueError("integration name must be a literal lowercase skill name")
        if spec["name"] in seen:
            raise ValueError("duplicate integration registry name: " + spec["name"])
        seen.add(spec["name"])
        upstream = spec.get("upstream", {})
        if not isinstance(upstream, dict):
            raise ValueError("upstream must be an object")
        content_hash = upstream.get("skill_sha256")
        if content_hash is not None and (not isinstance(content_hash, str) or re.fullmatch(r"[0-9a-f]{64}", content_hash) is None):
            raise ValueError("upstream skill_sha256 must be a lowercase SHA-256 string or null")
        if not isinstance(spec.get("dependencies", []), list):
            raise ValueError("dependencies must be a list")
        for dep in spec.get("dependencies", []):
            if not isinstance(dep, dict) or not isinstance(dep.get("distribution"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", dep["distribution"]):
                raise ValueError("dependency distribution must be a package metadata name")
        if not isinstance(spec.get("local_helpers", []), list):
            raise ValueError("local_helpers must be a list")
        for helper in spec.get("local_helpers", []):
            if not isinstance(helper, dict) or not isinstance(helper.get("path"), str) or not helper["path"].strip():
                raise ValueError("local helper must be an object with a nonempty path string")
            relative = Path(helper["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("helper paths must be relative and stay inside the skill")
        for field in ("executables", "artifact_requirements"):
            values = spec.get(field, [])
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{field} must be a list of nonempty strings")
    return registry


def load_registry(path=DEFAULT_REGISTRY):
    return validate_registry(load_json(path))


def discover(roots, registry=None, include_git=True, inspected_inputs=None):
    registry = load_registry() if registry is None else validate_registry(registry)
    roots = list(dict.fromkeys(Path(os.path.abspath(os.path.expanduser(str(p)))) for p in roots))
    warnings = []
    files = set()
    for root in roots:
        files.update(_candidate_files(root, warnings))
    if inspected_inputs is not None:
        inspected_inputs.update(files)
    expected = {spec["name"] for spec in registry["integrations"]}
    found = {name: [] for name in expected}
    for path in sorted(files):
        try:
            raw = path.read_bytes()
            name, version = frontmatter(raw.decode("utf-8"))
        except (OSError, UnicodeError) as exc:
            warnings.append({"path": str(path), "reason": "unreadable_skill", "detail": str(exc)})
            continue
        if name not in expected:
            warnings.append({"path": str(path), "reason": "unrecognized_or_invalid_frontmatter_name", "declared_name": name})
            continue
        found[name].append({
            "directory": str(path.parent), "skill_file": str(path),
            "skill_sha256": hashlib.sha256(raw).hexdigest(), "skill_bytes": len(raw),
            "declared_version": version,
            "declared_version_status": "available" if version is not None else "unavailable",
            "git": git_metadata(path.parent) if include_git else {"status": "not_checked", "commit": None},
        })
    results = []
    for spec in registry["integrations"]:
        matches = found[spec["name"]]
        pin = spec.get("upstream", {})
        expected_hash = pin.get("skill_sha256")
        for match in matches:
            match["pinned_skill_content"] = ("match" if match["skill_sha256"] == expected_hash else "diverged") if expected_hash else "unrecorded"
            match["warnings"] = []
            if match["pinned_skill_content"] == "diverged":
                match["warnings"].append("Installed SKILL.md differs from the reviewed pin; read its full contents and reassess compatibility.")
            match["helpers"] = []
            for helper in spec.get("local_helpers", []):
                path = Path(match["directory"]) / helper["path"]
                if inspected_inputs is not None:
                    inspected_inputs.add(path)
                available = _safe_under(path, Path(match["directory"])) and path.is_file()
                match["helpers"].append({**helper, "status": "present_not_executed" if available else "unavailable",
                                         "actual_path": str(path) if available else None})
        status = "located" if len(matches) == 1 else "ambiguous" if matches else "missing"
        results.append({
            "name": spec["name"], "status": status,
            "selected_skill_file": matches[0]["skill_file"] if status == "located" else None,
            "matches": matches, "upstream": pin,
            "dependencies": package_inventory(spec.get("dependencies", [])),
            "executables": [{"name": name, "path": shutil.which(name), "executed": False} for name in spec.get("executables", [])],
            "artifact_requirements": spec.get("artifact_requirements", []),
            "capability_verified": False,
            "next_action": "read_full_skill_and_check_task_prerequisites" if status == "located" else "use_core_fallback; resolve_local_path_before_delegation",
        })
    return {"schema_version": 1, "mode": "local_inventory_only", "roots": [str(p) for p in roots],
            "registry_revision": registry.get("registry_revision"), "integrations": results,
            "warnings": warnings,
            "limits": ["No skill, helper, training script, or package import was executed.",
                       "Package metadata and matching SKILL.md content do not establish runtime compatibility.",
                       "The SKILL.md hash covers that file only, not helper scripts or the full installation.",
                       "No network, automatic installation, or recursive search is performed."]}


def protect_scan_roots(output, roots):
    """Never write within scanned trees, including symlink aliases into them."""
    if output is None:
        return
    target = Path(os.path.abspath(os.path.expanduser(str(output))))
    for value in roots:
        root = Path(os.path.abspath(os.path.expanduser(str(value))))
        for candidate, boundary in ((target, root), (target.resolve(), root.resolve())):
            try:
                candidate.relative_to(boundary)
            except ValueError:
                continue
            raise ValueError("--output must be outside every supplied scan root")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path,
                        help="Explicit skill directory or containing directory; repeat for multiple roots")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--no-git", action="store_true", help="Skip bounded local Git metadata checks")
    parser.add_argument("--output", type=Path, help="Write JSON outside scan roots; must not alias any inspected input")
    args = parser.parse_args(argv)
    try:
        protect_scan_roots(args.output, args.root)
        inspected_inputs = {args.registry}
        report = discover(args.root, load_registry(args.registry), include_git=not args.no_git,
                          inspected_inputs=inspected_inputs)
        emit(report, args.output, inputs=inspected_inputs)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
