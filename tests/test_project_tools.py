"""Behavior checks for packaging, validation, and non-destructive installation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

PROJECT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(f"project_tool_{name}", PROJECT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


validator = load_tool("validate_project")
builder = load_tool("build_release")
installer = load_tool("install_skill")


def fixture(root):
    for relative in validator.REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "Fixture\n", encoding="utf-8")
    (root / "VERSION").write_text("0.1.0\n", encoding="utf-8")
    (root / "plugin.json").write_text('{"name":"train-opt-skills","version":"0.1.0"}\n', encoding="utf-8")
    skill = root / "skills" / "optimize-training"
    (skill / "SKILL.md").write_text(
        "---\nname: optimize-training\ndescription: >-\n  Diagnose slow training\n  and verify optimizations.\n---\n# Training\n",
        encoding="utf-8",
    )
    (skill / "scripts").mkdir()
    (skill / "scripts" / "example.py").write_text('print("example")\n', encoding="utf-8")
    (skill / "references").mkdir()
    (skill / "references" / "workflow.md").write_text("# Workflow\n", encoding="utf-8")
    return skill


class ProjectToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project"
        self.root.mkdir()
        self.skill = fixture(self.root)

    def test_valid_project_and_url_decoded_links(self):
        (self.root / "docs" / "with space.md").write_text("# Title\n", encoding="utf-8")
        (self.root / "README.md").write_text(
            "[local](docs/with%20space.md#title)\n"
            '[titled](<docs/with space.md> "A title")\n'
            "[ref]: docs/with%20space.md#title\n"
            "[external](https://example.org/unavailable.md)\n"
            "[mail](mailto:somebody@example.org)\n"
            "[anchor](#title)\n"
            "```markdown\n[illustrative](not-a-real-file.md)\n```\n",
            encoding="utf-8",
        )
        self.assertEqual(validator.validate_project(self.root), [])

    def test_broken_and_escaping_links_are_reported(self):
        (self.root / "README.md").write_text("[missing](missing.md)\n[outside](../outside.md)\n", encoding="utf-8")
        errors = "\n".join(validator.validate_project(self.root))
        self.assertIn("broken local link: missing.md", errors)
        self.assertIn("local link escapes project", errors)

    def test_frontmatter_name_description_and_line_limit(self):
        (self.skill / "SKILL.md").write_text(
            "---\nname: Other\ndescription: ''\n---\n" + "line\n" * 500, encoding="utf-8"
        )
        errors = "\n".join(validator.validate_project(self.root))
        self.assertIn("name must match directory", errors)
        self.assertIn("description must contain", errors)
        self.assertIn("500-line limit", errors)

    def test_invalid_json_and_mismatched_versions(self):
        (self.root / "plugin.json").write_text('{"version":"9.0.0"}', encoding="utf-8")
        (self.skill / "assets" / "experiment.json").write_text("{broken", encoding="utf-8")
        errors = "\n".join(validator.validate_project(self.root))
        self.assertIn("invalid JSON", errors)
        self.assertIn("version must match VERSION", errors)

    def test_installer_dry_run_install_and_no_overwrite(self):
        parent = self.root.parent / "new-parent" / "skills"
        expected = parent / "optimize-training"
        self.assertEqual(installer.install_skill(self.skill, parent, dry_run=True), expected)
        self.assertFalse(parent.exists())
        installer.install_skill(self.skill, parent)
        entry = expected / "SKILL.md"
        self.assertEqual(entry.read_bytes(), (self.skill / "SKILL.md").read_bytes())
        entry.write_text("User customization\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            installer.install_skill(self.skill, parent)
        self.assertEqual(entry.read_text(encoding="utf-8"), "User customization\n")

    def test_installer_rejects_target_inside_source(self):
        with self.assertRaisesRegex(ValueError, "inside the source"):
            installer.install_skill(self.skill, self.skill / "nested")
        self.assertFalse((self.skill / "nested").exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_installer_rejects_source_symlink_and_dangling_destination(self):
        parent = self.root.parent / "installed"
        parent.mkdir()
        (parent / "optimize-training").symlink_to(self.root.parent / "missing", target_is_directory=True)
        with self.assertRaises(FileExistsError):
            installer.install_skill(self.skill, parent)
        (parent / "optimize-training").unlink()
        (self.skill / "dangerous-link").symlink_to(self.root / "LICENSE")
        with self.assertRaisesRegex(ValueError, "source contains a symlink"):
            installer.install_skill(self.skill, parent)
        self.assertFalse((parent / "optimize-training").exists())

    def test_release_is_deterministic_and_manifests_are_valid(self):
        for folder in (".git", "__pycache__", ".venv", "dist"):
            path = self.root / folder
            path.mkdir()
            (path / "unwanted.txt").write_text("excluded", encoding="utf-8")
        (self.skill / "scripts" / "bad.pyc").write_bytes(b"excluded")
        paths = builder.build_archives(self.root)
        first = {path.name: path.read_bytes() for path in paths}
        first_checksums = (self.root / "dist" / "SHA256SUMS").read_bytes()
        os.utime(self.skill / "SKILL.md", (1800000000, 1800000000))
        builder.build_archives(self.root)
        self.assertEqual(first, {path.name: path.read_bytes() for path in paths})
        self.assertEqual(first_checksums, (self.root / "dist" / "SHA256SUMS").read_bytes())
        for path in paths:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                self.assertEqual(names, sorted(names))
                self.assertFalse(any("unwanted.txt" in name or name.endswith(".pyc") for name in names))
                self.assertFalse(any("/dist/" in name or "/.git/" in name for name in names))
                manifest_name = next(name for name in names if name.endswith("/RELEASE-MANIFEST.json"))
                prefix = manifest_name.rsplit("/", 1)[0]
                manifest = json.loads(archive.read(manifest_name))
                declared = {f'{prefix}/{item["path"]}' for item in manifest["files"]}
                self.assertEqual(declared, set(names) - {manifest_name})
                for item in manifest["files"]:
                    payload = archive.read(f'{prefix}/{item["path"]}')
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
                    self.assertEqual(len(payload), item["bytes"])

    def test_custom_output_directory_is_not_packaged_on_rebuild(self):
        output = self.root / "artifacts"
        paths = builder.build_archives(self.root, output)
        before = paths[0].read_bytes()
        builder.build_archives(self.root, output)
        self.assertEqual(before, paths[0].read_bytes())
        with zipfile.ZipFile(paths[0]) as archive:
            self.assertFalse(any("/artifacts/" in name for name in archive.namelist()))


if __name__ == "__main__":
    unittest.main()
