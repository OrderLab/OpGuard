"""Check installation defaults without a CUDA toolchain."""
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import Mock, patch

SETUP = Path(__file__).resolve().parents[1] / 'setup.py'


class BuildDefaultsTests(unittest.TestCase):
    def run_setup(self, env, system):
        setuptools = types.ModuleType('setuptools')
        setuptools.setup = Mock()
        setuptools.Extension = Mock()
        with patch.dict(os.environ, env, clear=True), patch.dict(
            sys.modules, {'setuptools': setuptools, 'torch': None}
        ), patch('platform.system', return_value=system):
            runpy.run_path(str(SETUP))
        return setuptools.setup

    def test_default_rejects_unsupported_platform(self):
        with self.assertRaisesRegex(RuntimeError, 'Linux only'):
            self.run_setup({}, 'Darwin')

    def test_default_requires_training_torch(self):
        with self.assertRaisesRegex(RuntimeError, '--no-build-isolation'):
            self.run_setup({}, 'Linux')

    def test_cpu_only_requires_explicit_opt_out_of_both_extensions(self):
        setup = self.run_setup({'OPGUARD_TRACING_BUILD': '0',
                                'OPGUARD_CALLSITE_TRACER': '0'}, 'Darwin')
        self.assertEqual(setup.call_args.kwargs['ext_modules'], [])
        for flag in ('OPGUARD_TRACING_BUILD', 'OPGUARD_CALLSITE_TRACER'):
            with self.subTest(flag=flag), self.assertRaisesRegex(RuntimeError, 'Linux only'):
                self.run_setup({flag: '0'}, 'Darwin')
