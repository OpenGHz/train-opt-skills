"""Optional discovery must not turn unverified local files into runnable skills."""
from __future__ import annotations

import builtins
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
SKILL = PROJECT / "skills" / "optimize-training"


def load_discovery():
    spec = importlib.util.spec_from_file_location("integration_discovery", SKILL / "scripts" / "discover_integrations.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with mock.patch.object(sys, "path", [str(SKILL / "scripts"), *sys.path]):
        spec.loader.exec_module(module)
    return module


discovery = load_discovery()


def minimal_registry():
    return {"schema_version": 1, "integrations": [{"name": "profile-model", "dependencies": []}]}


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def create(self, relative, name="profile-model", body="An entry.", version=None):
        path = self.root / relative / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = f'metadata:\n  version: "{version}"\n' if version else ""
        path.write_text(f"---\nname: {name}\ndescription: A fixture\n{metadata}---\n{body}\n", encoding="utf-8")
        return path

    def report(self, roots=None, registry=None):
        return discovery.discover(roots or [self.root], registry or minimal_registry(), include_git=False)

    def test_missing_reports_unknown_and_core_fallback(self):
        report = self.report()
        entry = report["integrations"][0]
        self.assertEqual(entry["status"], "missing")
        self.assertIsNone(entry["selected_skill_file"])
        self.assertFalse(entry["capability_verified"])
        self.assertIn("fallback", entry["next_action"])

    def test_generated_directory_name_and_exact_full_content_hash(self):
        path = self.create("skill-123", body="full content\nfinal line", version="1.4")
        entry = self.report()["integrations"][0]
        self.assertEqual(entry["status"], "located")
        self.assertEqual(entry["selected_skill_file"], str(path))
        match = entry["matches"][0]
        self.assertEqual(match["directory"], str(path.parent))
        self.assertEqual(match["skill_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(match["declared_version"], "1.4")
        self.assertEqual(match["git"]["status"], "not_checked")

    def test_missing_version_is_not_inferred_from_pin(self):
        self.create("profile-model")
        registry = minimal_registry()
        registry["integrations"][0]["upstream"] = {"repository_commit": "a" * 40, "skill_sha256": "0" * 64}
        match = self.report(registry=registry)["integrations"][0]["matches"][0]
        self.assertIsNone(match["declared_version"])
        self.assertEqual(match["declared_version_status"], "unavailable")
        self.assertEqual(match["pinned_skill_content"], "diverged")
        self.assertTrue(match["warnings"])

    def test_two_distinct_copies_are_ambiguous_even_when_identical(self):
        self.create("one")
        self.create("two")
        entry = self.report()["integrations"][0]
        self.assertEqual(entry["status"], "ambiguous")
        self.assertIsNone(entry["selected_skill_file"])
        self.assertEqual(len(entry["matches"]), 2)

    def test_overlapping_roots_deduplicate_same_file(self):
        path = self.create("skills/generated")
        entry = self.report([self.root, self.root / "skills", path.parent, self.root])["integrations"][0]
        self.assertEqual(entry["status"], "located")
        self.assertEqual(len(entry["matches"]), 1)

    def test_frontmatter_name_mismatch_is_not_matched_by_directory(self):
        self.create("profile-model", name="unrelated")
        report = self.report()
        self.assertEqual(report["integrations"][0]["status"], "missing")
        self.assertEqual(report["warnings"][0]["declared_name"], "unrelated")

    def test_missing_or_duplicate_frontmatter_names_are_invalid(self):
        self.assertEqual(discovery.frontmatter("# name: profile-model"), (None, None))
        self.assertEqual(discovery.frontmatter("---\nname: profile-model\nname: profile-model\n---"), (None, None))
        self.assertEqual(discovery.frontmatter("---\nname: !!python/object:unsafe\n---"), (None, None))
        self.assertEqual(discovery.frontmatter("---\nname: 'profile-model'\n---")[0], "profile-model")

    def test_does_not_recurse_into_arbitrary_nested_directories(self):
        self.create("unrelated/deep")
        self.create("skills/deep/nested")
        self.assertEqual(self.report()["integrations"][0]["status"], "missing")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_symlink_child_skill_and_root_do_not_escape_search(self):
        outside = self.create("outside")
        search = self.root / "search"
        search.mkdir()
        (search / "linked").symlink_to(outside.parent, target_is_directory=True)
        (search / "SKILL.md").symlink_to(outside)
        entry = self.report([search])["integrations"][0]
        self.assertEqual(entry["status"], "missing")
        self.assertEqual(self.report([search / "linked"])["integrations"][0]["status"], "missing")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_symlink_optional_skills_directory_and_ancestor_are_not_followed(self):
        outside = self.create("outside/profile-model")
        search = self.root / "search"
        search.mkdir()
        (search / "skills").symlink_to(outside.parent.parent, target_is_directory=True)
        self.assertEqual(self.report([search])["integrations"][0]["status"], "missing")
        self.assertEqual(self.report([search / "skills" / "profile-model"])["integrations"][0]["status"], "missing")

    def test_package_metadata_does_not_import_torch_or_pandas(self):
        self.create("profile-model")
        registry = minimal_registry()
        registry["integrations"][0]["dependencies"] = [{"distribution": "torch"}, {"distribution": "pandas"}]
        original_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "pandas", "hta"):
                raise AssertionError("Discovery imported a runtime dependency")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guard), mock.patch.object(
                importlib.metadata, "version", side_effect=["2.example", importlib.metadata.PackageNotFoundError()]):
            dependencies = self.report(registry=registry)["integrations"][0]["dependencies"]
        self.assertEqual(dependencies[0]["status"], "metadata_present")
        self.assertEqual(dependencies[1]["status"], "metadata_absent")
        self.assertFalse(any(item["runtime_import_tested"] for item in dependencies))

    def test_helper_only_reported_not_executed(self):
        path = self.create("profile-model")
        helper = path.parent / "scripts" / "helper.py"
        helper.parent.mkdir()
        helper.write_text("raise RuntimeError('must not run')\n", encoding="utf-8")
        registry = minimal_registry()
        registry["integrations"][0]["local_helpers"] = [{"path": "scripts/helper.py"}]
        entry = self.report(registry=registry)["integrations"][0]
        self.assertEqual(entry["matches"][0]["helpers"][0]["status"], "present_not_executed")
        self.assertFalse(entry["capability_verified"])

    def test_pin_file_match_is_separate_from_repository_commit(self):
        path = self.create("profile-model")
        registry = minimal_registry()
        registry["integrations"][0]["upstream"] = {"repository_commit": "f" * 40, "skill_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with mock.patch.object(discovery, "git_metadata", return_value={"status": "available", "commit": "a" * 40}):
            report = discovery.discover([self.root], registry)
        match = report["integrations"][0]["matches"][0]
        self.assertEqual(match["pinned_skill_content"], "match")
        self.assertEqual(match["git"]["commit"], "a" * 40)

    def test_git_timeout_reports_unknown(self):
        with mock.patch.object(discovery.shutil, "which", return_value="/usr/bin/git"), mock.patch.object(
                discovery.subprocess, "run", side_effect=subprocess.TimeoutExpired("git", 2)) as runner:
            report = discovery.git_metadata(self.root)
        self.assertEqual(report["status"], "unavailable")
        self.assertIsNone(report["commit"])
        self.assertEqual(runner.call_args.kwargs["timeout"], 2)
        self.assertNotIn("shell", runner.call_args.kwargs)

    def test_registry_rejects_escaping_helper_and_duplicate_name(self):
        registry = minimal_registry()
        registry["integrations"][0]["local_helpers"] = [{"path": "../outside.py"}]
        path = self.root / "registry.json"
        path.write_text(json.dumps(registry), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inside the skill"):
            discovery.load_registry(path)
        registry = minimal_registry()
        registry["integrations"].append(dict(registry["integrations"][0]))
        path.write_text(json.dumps(registry), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            discovery.load_registry(path)

    def test_default_registry_has_distinct_commit_and_blob_pins(self):
        registry = discovery.load_registry()
        for item in registry["integrations"]:
            pin = item["upstream"]
            if item["name"] == "nsight-systems":
                self.assertIsNone(pin["repository_commit"])
                continue
            self.assertEqual(len(pin["skill_sha256"]), 64)
            self.assertEqual(len(pin["git_blob_sha"]), 40)
            self.assertNotEqual(pin["git_blob_sha"], pin["repository_commit"])

    def test_cli_missing_and_invalid_registry_exit_codes(self):
        command = [sys.executable, str(SKILL / "scripts" / "discover_integrations.py"), "--root", str(self.root), "--no-git"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertTrue(all(item["status"] == "missing" for item in report["integrations"]))
        bad = self.root / "bad.json"
        bad.write_text("not JSON", encoding="utf-8")
        completed = subprocess.run(command + ["--registry", str(bad)], capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 2)

    def output_fixture(self):
        skill = self.create("scan/profile-model")
        helper = skill.parent / "scripts" / "helper.py"
        helper.parent.mkdir()
        helper.write_text("# Do not overwrite this helper\n", encoding="utf-8")
        unknown = self.create("scan/unknown", name="unknown-skill")
        registry = minimal_registry()
        registry["integrations"][0]["local_helpers"] = [{"path": "scripts/helper.py"}]
        registry_path = self.root / "registry.json"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        command = [sys.executable, str(SKILL / "scripts" / "discover_integrations.py"),
                   "--root", str(self.root / "scan"), "--registry", str(registry_path), "--no-git"]
        return command, registry_path, skill, helper, unknown

    def assert_output_rejected_without_mutation(self, command, output, inputs):
        originals = {path: path.read_bytes() for path in inputs}
        completed = subprocess.run(command + ["--output", str(output)], capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 2, completed.stderr)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content, str(path))
        self.assertFalse(list(self.root.rglob(".training-report-*")))

    def test_output_rejects_registry_skill_helper_and_unknown_entry(self):
        command, *inputs = self.output_fixture()
        for target in inputs:
            with self.subTest(target=target.name):
                self.assert_output_rejected_without_mutation(command, target, inputs)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_output_rejects_symlink_aliases_of_all_inspected_inputs(self):
        command, *inputs = self.output_fixture()
        for index, target in enumerate(inputs):
            alias = self.root / f"symlink-{index}"
            alias.symlink_to(target)
            with self.subTest(target=str(target)):
                self.assert_output_rejected_without_mutation(command, alias, inputs)
                self.assertTrue(alias.is_symlink())

    @unittest.skipUnless(hasattr(os, "link"), "hardlinks unsupported")
    def test_output_rejects_hardlink_aliases_of_all_inspected_inputs(self):
        command, *inputs = self.output_fixture()
        for index, target in enumerate(inputs):
            alias = self.root / f"hardlink-{index}"
            os.link(target, alias)
            with self.subTest(target=str(target)):
                self.assert_output_rejected_without_mutation(command, alias, inputs)
                self.assertTrue(os.path.samefile(target, alias))

    def test_output_rejects_new_file_anywhere_under_scan_root(self):
        command, *inputs = self.output_fixture()
        target = self.root / "scan" / "new.json"
        self.assert_output_rejected_without_mutation(command, target, inputs)
        self.assertFalse(target.exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_output_rejects_directory_alias_into_scan_root_and_link_outside(self):
        command, *inputs = self.output_fixture()
        alias = self.root / "scan-alias"
        alias.symlink_to(self.root / "scan", target_is_directory=True)
        self.assert_output_rejected_without_mutation(command, alias / "new.json", inputs)
        self.assertFalse((self.root / "scan" / "new.json").exists())
        outside = self.root / "outside.json"
        outside.write_text("preserve\n", encoding="utf-8")
        link_outside = self.root / "scan" / "outside-link"
        link_outside.symlink_to(outside)
        self.assert_output_rejected_without_mutation(command, link_outside, inputs + [outside])
        self.assertTrue(link_outside.is_symlink())

    def test_output_normal_file_outside_roots_is_atomic_and_preserves_inputs(self):
        command, *inputs = self.output_fixture()
        originals = {path: path.read_bytes() for path in inputs}
        target = self.root / "inventory.json"
        target.write_text("old report\n", encoding="utf-8")
        completed = subprocess.run(command + ["--output", str(target)], capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(target.read_text())["mode"], "local_inventory_only")
        self.assertEqual(completed.stdout, "")
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertFalse(list(self.root.rglob(".training-report-*")))

    def test_malformed_registry_never_creates_or_replaces_output(self):
        command, registry, *inputs = self.output_fixture()
        registry.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        old = self.root / "old.json"
        old.write_text("preserve existing report\n", encoding="utf-8")
        new = self.root / "new.json"
        for output in (old, new):
            self.assert_output_rejected_without_mutation(command, output, [registry, old, *inputs])
        self.assertFalse(new.exists())

    def test_malformed_nested_registry_returns_exit_two_without_traceback_or_writes(self):
        command, registry_path, *inputs = self.output_fixture()
        target = self.root / "existing-report.json"
        target.write_text("preserve report\n", encoding="utf-8")
        malformed = [
            ("upstream", []), ("upstream", "not an object"),
            ("upstream", {"skill_sha256": []}),
            ("dependencies", {}), ("dependencies", ["torch"]),
            ("dependencies", [{"distribution": []}]),
            ("local_helpers", "scripts/helper.py"), ("local_helpers", [None]),
            ("local_helpers", [{"path": []}]),
            ("executables", "nsys"), ("executables", [None]),
            ("artifact_requirements", {}),
        ]
        for field, value in malformed:
            registry = minimal_registry()
            registry["integrations"][0][field] = value
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            with self.subTest(field=field, value=value):
                completed = subprocess.run(command + ["--output", str(target)], capture_output=True, text=True, timeout=10)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertIn("error:", completed.stderr)
                self.assertNotIn("Traceback", completed.stderr)
                self.assertEqual(target.read_text(), "preserve report\n")
                self.assertEqual(json.loads(registry_path.read_text()), registry)


if __name__ == "__main__":
    unittest.main()
