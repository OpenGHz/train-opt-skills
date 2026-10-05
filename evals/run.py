#!/usr/bin/env python3
"""Portable artifact evaluation runner; Python standard library only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CASES = {item["id"]: item for item in json.loads((HERE / "cases.json").read_text())["cases"]}


def write_json(path, obj):
    if path.is_symlink():
        raise ValueError(f"refusing to write through symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def safe_destination(path):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {path}")
    resolved = path.resolve()
    if resolved == PROJECT or PROJECT in resolved.parents:
        raise ValueError("evaluation output must be outside the source project")
    return resolved


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive_timeout(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be finite and positive")
    return value


def eval_directory(workdir):
    directory = workdir / ".eval"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(".eval must be a real directory inside the workspace")
    for path in directory.iterdir():
        if path.is_symlink():
            raise ValueError(f"refusing symlink in evaluation metadata: {path.name}")
    return directory


def task_text(case, variant):
    fixture = HERE / CASES[case]["fixture"]
    task = fixture.joinpath("prompt.md").read_text(encoding="utf-8")
    task += ("\nWork only in this task directory. Finish by writing response.json matching response.schema.json. "
             f"Use schema_version=1 and case_id={case!r}. status is fixed, incomplete, or rejected. "
             "summary must explain what you did and what the evidence supports; unverified lists remaining limitations. "
             "Do not claim GPU or production-training validation from CPU fixtures. "
             "Do not read the source evaluation harness, expected outputs, or other attempts.\n")
    if variant == "treatment":
        task += "\nRead and use the supplied core skill at _skill/optimize-training/SKILL.md.\n"
    return task


def expected_immutable(case, variant):
    fixture = HERE / CASES[case]["fixture"]
    expected = {p.name: sha256(p) for p in fixture.iterdir() if p.is_file() and p.name not in ["pipeline.py", "prompt.md"]}
    expected["response.schema.json"] = sha256(HERE / "response.schema.json")
    expected["TASK.md"] = hashlib.sha256(task_text(case, variant).encode("utf-8")).hexdigest()
    return expected


def prepare(case, variant, output):
    output = safe_destination(output)
    fixture = HERE / CASES[case]["fixture"]
    output.mkdir(parents=True)
    for source in fixture.iterdir():
        if source.name != "prompt.md":
            shutil.copy2(source, output / source.name)
    shutil.copy2(HERE / "response.schema.json", output / "response.schema.json")
    if variant == "treatment":
        source_skill = PROJECT / "skills" / "optimize-training"
        if any(p.is_symlink() for p in source_skill.rglob("*")):
            raise ValueError("skill source contains a symlink")
        shutil.copytree(source_skill, output / "_skill" / "optimize-training", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (output / "TASK.md").write_text(task_text(case, variant), encoding="utf-8")
    immutable = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != "pipeline.py"}
    metadata = {"schema_version": 1, "case_id": case, "variant": variant,
                "prepared_at": timestamp(), "immutable_inputs": immutable,
                "fixture_sha256": {p.name: sha256(p) for p in fixture.iterdir() if p.is_file()},
                "execution": "not_run", "isolation": "fixture_copy_only; not an OS sandbox"}
    if variant == "treatment":
        skill = output / "_skill" / "optimize-training"
        metadata["skill_sha256"] = {str(p.relative_to(skill)): sha256(p) for p in skill.rglob("*") if p.is_file()}
    write_json(output / ".eval" / "manifest.json", metadata)
    return output


def check_response(obj, case):
    return (isinstance(obj, dict) and type(obj.get("schema_version")) is int and obj["schema_version"] == 1
            and obj.get("case_id") == case and obj.get("status") in ["fixed", "incomplete", "rejected"]
            and isinstance(obj.get("summary"), str) and bool(obj["summary"].strip())
            and isinstance(obj.get("unverified"), list) and all(isinstance(v, str) for v in obj["unverified"]))


def score(case, workdir, checker_timeout=15):
    positive_timeout(checker_timeout)
    workdir = Path(workdir).resolve()
    eval_directory(workdir)
    checks = []
    manifest = json.loads((workdir / ".eval" / "manifest.json").read_text())
    if manifest.get("case_id") != case:
        raise ValueError("case does not match prepared workspace")
    if manifest.get("variant") not in ["baseline", "treatment"]:
        raise ValueError("invalid prepared variant")
    expected = expected_immutable(case, manifest["variant"])
    checks.append({"name": "manifest_inputs_match_harness", "passed": manifest.get("immutable_inputs") == expected})
    immutable_ok = all(not (workdir / name).is_symlink() and (workdir / name).is_file() and sha256(workdir / name) == digest
                       for name, digest in expected.items())
    checks.append({"name": "immutable_inputs_preserved", "passed": immutable_ok})
    try:
        response = json.loads((workdir / "response.json").read_text())
        response_ok = check_response(response, case)
        response_status = "present" if response_ok else "invalid"
    except (OSError, ValueError):
        response_ok = False
        response_status = "missing_or_invalid"
    checks.append({"name": "response_schema", "passed": response_ok})
    if response_ok:
        try:
            result = subprocess.run([sys.executable, str(HERE / "checkers.py"), case, str(workdir)],
                                    cwd=workdir, capture_output=True, text=True, timeout=checker_timeout, check=False)
            extra = json.loads(result.stdout)
            if result.returncode != 0 or not isinstance(extra, list) or not extra:
                raise ValueError("checker returned invalid result")
            checks.extend(extra)
        except (subprocess.TimeoutExpired, ValueError) as exc:
            checks.append({"name": "artifact_checker", "passed": False, "detail": str(exc)})
    report = {"schema_version": 1, "case_id": case, "variant": manifest["variant"],
              "scored_at": timestamp(), "response_status": response_status, "checks": checks,
              "passed": response_ok and all(c["passed"] is True for c in checks),
              "scope": "Small deterministic CPU artifact checks; not scientific or production training acceptance."}
    runtime_path = workdir / ".eval" / "runtime.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        report["runtime_status"] = runtime["status"]
        if runtime["status"] != "completed" or runtime["returncode"] != 0:
            report["passed"] = False
    write_json(workdir / ".eval" / "score.json", report)
    return report


def execute(command, workdir, variant, timeout):
    positive_timeout(timeout)
    eval_directory(workdir)
    env = os.environ.copy()
    env.pop("TASK_SKILL_PATH", None)
    env.update(TASK_PROMPT_PATH=str(workdir / "TASK.md"), TASK_WORKDIR=str(workdir), EXEC_VARIANT=variant)
    if variant == "treatment":
        env["TASK_SKILL_PATH"] = str(workdir / "_skill" / "optimize-training")
    metadata = {"schema_version": 1, "started_at": timestamp(), "runtime_argv": command,
                "timeout_seconds": timeout, "status": "launch_failed", "returncode": None}
    stdout, stderr = "", ""
    try:
        process = subprocess.Popen(command, cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=(os.name == "posix"))
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            metadata["status"] = "completed"
        except BaseException as exc:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            stdout, stderr = process.communicate()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            metadata["status"] = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "runtime_error"
            if metadata["status"] == "runtime_error":
                stderr += f"\nRuntime communication error: {type(exc).__name__}: {exc}\n"
        metadata["returncode"] = process.returncode
    except OSError as exc:
        stderr = str(exc)
    metadata["finished_at"] = timestamp()
    eval_directory(workdir)
    (workdir / ".eval" / "stdout.txt").write_text(stdout, encoding="utf-8")
    (workdir / ".eval" / "stderr.txt").write_text(stderr, encoding="utf-8")
    write_json(workdir / ".eval" / "runtime.json", metadata)
    manifest_path = workdir / ".eval" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["execution"] = metadata["status"]
    write_json(manifest_path, manifest)
    return metadata


def run_suite(command, variant, output, timeout, cases):
    positive_timeout(timeout)
    output = safe_destination(output)
    output.mkdir(parents=True)
    reports = []
    variants = ["baseline", "treatment"] if variant == "both" else [variant]
    for case in cases:
        for mode in variants:
            workdir = prepare(case, mode, output / case / mode)
            execute(command, workdir, mode, timeout)
            reports.append(score(case, workdir))
    summary = {"schema_version": 1, "kind": "runtime_evaluation", "attempts": reports,
               "passed_attempts": sum(r["passed"] for r in reports), "total_attempts": len(reports),
               "limitations": ["Fixture isolation is not a filesystem sandbox.", "Small synthetic cases do not establish GPU speedup or broad agent success rate."]}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--case", choices=CASES, required=True)
    prep.add_argument("--variant", choices=["baseline", "treatment"], required=True)
    prep.add_argument("--output", type=Path, required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--case", choices=CASES, required=True)
    scoring.add_argument("--workdir", type=Path, required=True)
    scoring.add_argument("--checker-timeout", type=float, default=15)
    run = commands.add_parser("run")
    run.add_argument("--runtime-json", required=True, help="JSON array of command and arguments; no shell evaluation")
    run.add_argument("--variant", choices=["baseline", "treatment", "both"], default="both")
    run.add_argument("--case", choices=CASES, action="append")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            print(prepare(args.case, args.variant, args.output))
            return 0
        if args.action == "score":
            result = score(args.case, args.workdir, args.checker_timeout)
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] else 1
        command = json.loads(args.runtime_json)
        if not isinstance(command, list) or not command or not all(isinstance(s, str) and s for s in command):
            raise ValueError("runtime-json must be a nonempty JSON array of strings")
        positive_timeout(args.timeout)
        result = run_suite(command, args.variant, args.output, args.timeout, args.case or list(CASES))
        print(json.dumps({"passed_attempts": result["passed_attempts"], "total_attempts": result["total_attempts"]}))
        return 0 if result["passed_attempts"] == result["total_attempts"] else 1
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
