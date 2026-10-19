"""Portable launcher adaptation must not depend on the original host layout."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'examples/independent-training-alignment/prepare_workload.py'
spec = importlib.util.spec_from_file_location('prepare_workload', MODULE)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PortableLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = ('launch.sh', 'stack_a/launch.sh', 'stack_b/launch.sh', 'stack_b/run_training.sh')
        for name in self.names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            prefix = 'export PATH="/example/training-env/bin:$PATH"\n' if name == 'stack_b/launch.sh' else ''
            path.write_text('#!/bin/bash\n' + prefix + '/example/training-env/bin/python train.py "$@"\n')

    def test_adapts_verified_archive_without_host_specific_paths(self):
        self.assertEqual(prepare.portable_launchers(self.root), list(self.names))
        for name in self.names:
            content = (self.root / name).read_text()
            self.assertNotIn('/example/training-env', content)
            self.assertIn('python3', content)
        self.assertIn('command -v', (self.root / 'stack_b/launch.sh').read_text())

    def test_ambiguous_interpreters_fail_before_modifying_files(self):
        path = self.root / 'launch.sh'
        path.write_text(path.read_text() + '/another/env/bin/python3 train.py\n')
        before = {name: (self.root / name).read_bytes() for name in self.names}
        with self.assertRaisesRegex(ValueError, 'one archived Python'):
            prepare.portable_launchers(self.root)
        self.assertEqual(before, {name: (self.root / name).read_bytes() for name in self.names})


if __name__ == '__main__':
    unittest.main()
