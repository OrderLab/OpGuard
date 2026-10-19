import ast
import json
from pathlib import Path
import tempfile
import sys
import unittest

from opguard_harness.input_bundle import export_inputs, verify_inputs


class InputBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "workload.py"
        self.source.write_text("value = 1\n")
        self.output = self.root / "input-bundle"

    def test_only_allowlisted_files_are_exported_without_origin_paths(self):
        answers = self.root / "answers"
        answers.mkdir()
        (answers / "diagnosis.json").write_text('{"cause": "secret"}')
        manifest = export_inputs(self.output, {"task/workload.py": self.source})
        self.assertEqual(set(manifest["files"]), {"task/workload.py"})
        self.assertNotIn(str(self.root), (self.output / "manifest.json").read_text())
        self.assertEqual(verify_inputs(self.output), manifest)
        self.assertFalse((self.output / "answers").exists())

    def test_cannot_merge_with_old_captures(self):
        self.output.mkdir()
        stale = self.output / "old-trace.json"
        stale.write_text("embedded repair")
        with self.assertRaises(FileExistsError):
            export_inputs(self.output, {"workload.py": self.source})
        self.assertEqual(stale.read_text(), "embedded repair")

    def test_answers_history_and_path_escapes_are_rejected_before_copy(self):
        for name in ("../outside.py", "/absolute.py", "a/../outside.py", "a//x.py",
                     "answers/fix.py", ".git/config", "x/__pycache__/x.pyc", "manifest.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                export_inputs(self.output, {name: self.source})
            self.assertFalse(self.output.exists())
        hidden = self.root / "answers" / "fix.py"
        hidden.parent.mkdir()
        hidden.write_text("repair = True")
        with self.assertRaises(ValueError):
            export_inputs(self.output, {"innocent.py": hidden})

    def test_symlinked_input_cannot_import_private_material(self):
        link = self.root / "linked.py"
        link.symlink_to(self.source)
        with self.assertRaises(ValueError):
            export_inputs(self.output, {"workload.py": link})
        directory_link = self.root / "alias"
        directory_link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            export_inputs(self.output, {"workload.py": directory_link / "workload.py"})

    @unittest.skipUnless(sys.platform == "darwin", "macOS system directory aliases")
    def test_macos_system_tmp_alias_is_allowed(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            source = root / "workload.py"
            source.write_text("value = 1\n")
            output = root / "bundle"
            manifest = export_inputs(output, {"workload.py": source})
            self.assertEqual(verify_inputs(output), manifest)
            private = root / "answers"
            private.mkdir()
            (private / "answer.py").write_text("repair = True\n")
            with self.assertRaises(ValueError):
                export_inputs(root / "other", {"workload.py": private / "answer.py"})

    def test_inventory_detects_new_diagnostics_mutation_and_links(self):
        export_inputs(self.output, {"workload.py": self.source})
        extra = self.output / "diagnosis.json"
        extra.write_text("secret")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_inputs(self.output)
        extra.unlink()
        workload = self.output / "workload.py"
        workload.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Changed"):
            verify_inputs(self.output)
        workload.unlink()
        workload.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            verify_inputs(self.output)

    def test_diagnostic_workloads_do_not_supply_known_repair_controls(self):
        templates = Path(__file__).resolve().parents[1] / "reproductions/table4/inputs"
        for path in templates.glob("workload_*.py"):
            source = path.read_text()
            strings = {n.value for n in ast.walk(ast.parse(source))
                       if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            with self.subTest(path=path.name):
                self.assertFalse(strings & {"--fix", "--reset", "--clear-rope", "--exact-grad"})
                self.assertNotIn("PR #5606", source)
                self.assertNotIn("issues/", source)
                self.assertNotIn("op.reset_cache()", source)
                self.assertNotIn("model.rope_deltas = None", source)
                self.assertNotIn("wait_stream(optimizer.reduction_stream)", source)


if __name__ == "__main__":
    unittest.main()
