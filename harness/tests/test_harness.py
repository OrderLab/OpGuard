import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from opguard_harness.engine import Harness
from opguard_harness.spec import validate
from opguard_harness.tools import dispatch

FAKE = r'''
import json, pathlib, sys
phase = sys.argv[1]
r = json.loads(pathlib.Path(sys.argv[3]).read_text())
p = pathlib.Path(r['output']); p.mkdir()
if phase in ('discover', 'trace'):
    result = dict(output=str(p), phase=phase, ranks=r['workload']['ranks'])
elif phase == 'prepare':
    config = p / 'capture.yaml'; config.write_text('instrumentations: []')
    result = dict(config=str(config), config_sha256='test', sources={'test.py': 'test'})
else:
    reports = []
    for rank in r['ranks']:
        report = p / (str(rank) + '.json.gz'); report.write_bytes(b'report'); reports.append(str(report))
    result = dict(reports=reports, ranks=r['ranks'], verdict='not_evaluated')
print(json.dumps(dict(protocol_version=1, ok=True, result=result)))
'''


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        backend = self.root / 'backend.py'
        backend.write_text(FAKE)
        self.spec = {'backend': [sys.executable, str(backend)], 'local_root': str(self.root / 'ssd')}
        for name in ('baseline', 'candidate'):
            cwd = self.root / name
            cwd.mkdir()
            (cwd / 'train.py').write_text('print("workload")')
            self.spec[name] = dict(cwd=str(cwd), script='train.py', launcher=[sys.executable])
        self.harness = Harness(self.root / 'experiments')

    def test_full_workflow_and_idempotent_completion(self):
        experiment = self.harness.create(self.spec)['id']
        for _ in range(7):
            state = self.harness.advance(experiment)
        self.assertEqual(state['status'], 'succeeded')
        self.assertTrue(self.harness.results(experiment)['complete'])
        self.assertEqual(self.harness.results(experiment)['verdict'], 'not_evaluated')
        self.assertEqual(self.harness.advance(experiment), state)
        self.assertEqual(len(list((self.root / 'experiments' / experiment).glob('*/request.json'))), 7)

    def test_failure_budget_and_unique_retry_outputs(self):
        self.spec['backend'] = [sys.executable, '-c', 'raise SystemExit(7)']
        experiment = self.harness.create(self.spec)['id']
        for _ in range(2):
            state = self.harness.advance(experiment)
            self.assertEqual(state['status'], 'failed')
        attempts = state['stages'][0]['attempts']
        self.assertNotEqual(attempts[0]['directory'], attempts[1]['directory'])
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            self.harness.advance(experiment)
        self.assertFalse(self.harness.results(experiment)['complete'])

    def test_malformed_success_does_not_advance(self):
        self.spec['backend'] = [sys.executable, '-c', 'print(\'{"protocol_version":1,"ok":true,"result":{}}\')']
        experiment = self.harness.create(self.spec)['id']
        state = self.harness.advance(experiment)
        self.assertEqual(state['next'], {'side': 'baseline', 'action': 'discover'})
        self.assertEqual(state['status'], 'failed')

    def test_independent_sessions_and_refinement(self):
        first = self.harness.create(self.spec)['id']
        second = self.harness.refine(first, self.spec)['id']
        self.harness.advance(second)
        self.assertEqual(self.harness.status(first)['status'], 'pending')
        self.assertEqual(self.harness.store.load(second)['parent'], first)

    def test_lock_prevents_duplicate_launch(self):
        experiment = self.harness.create(self.spec)['id']
        with self.harness.store.locked(experiment):
            with self.assertRaisesRegex(RuntimeError, 'another process'):
                self.harness.advance(experiment)

    def test_interrupted_run_not_retried(self):
        experiment = self.harness.create(self.spec)['id']
        record = self.harness.store.load(experiment)
        record['stages'][0]['status'] = 'running'
        self.harness.store.save(record)
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.harness.advance(experiment)

    def test_invalid_specs(self):
        for change in (
            lambda s: s.update(unrecognized=True),
            lambda s: s.update(timeout_seconds=float('nan')),
            lambda s: s['baseline'].update(launcher='python train.py'),
            lambda s: s['baseline'].update(ranks=[0, 0]),
            lambda s: s['baseline'].update(env={'OPGUARD_AUTOSTART': '1'}),
            lambda s: s['candidate'].update(cwd=s['baseline']['cwd']),
            lambda s: s['candidate'].update(ranks=[0, 1]),
        ):
            spec = copy.deepcopy(self.spec)
            change(spec)
            with self.assertRaises(ValueError):
                validate(spec)

    def test_tool_contract(self):
        with self.assertRaises(ValueError):
            dispatch(self.root, 'execute_shell', {})
        with self.assertRaises(ValueError):
            dispatch(self.root, 'create_experiment', {'wrong': self.spec})
        with self.assertRaises(ValueError):
            self.harness.status('../../other')

    def test_cli_schema_and_error_envelope(self):
        proc = subprocess.run([sys.executable, '-m', 'opguard_harness', 'tools'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        names = {tool['name'] for tool in json.loads(proc.stdout)['tools']}
        self.assertTrue({'review_pair', 'difference_candidates', 'render_alignment'} <= names)
        proc = subprocess.run([sys.executable, '-m', 'opguard_harness', '--root', str(self.root / 'cli'),
                               'call', 'create_experiment'], input='{}', capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertFalse(json.loads(proc.stdout)['ok'])


if __name__ == '__main__':
    unittest.main()


class RecoveryTests(unittest.TestCase):
    setUp = HarnessTests.setUp

    def test_recover_adopts_completed_backend_result(self):
        experiment = self.harness.create(self.spec)['id']
        self.harness.advance(experiment)
        record = self.harness.store.load(experiment)
        stage = record['stages'][0]
        stage['status'] = 'running'
        stage['attempts'][-1].pop('pid')
        stage['attempts'][-1]['status'] = 'running'
        stage.pop('result')
        record['status'] = 'running'
        self.harness.store.save(record)
        state = self.harness.recover(experiment)
        self.assertEqual(state['next']['action'], 'prepare')
        self.assertEqual(len(state['stages'][0]['attempts']), 1)

    def test_refine_nonexistent_parent_creates_nothing(self):
        with self.assertRaises(FileNotFoundError):
            self.harness.refine('0' * 32, self.spec)
        self.assertEqual(list((self.root / 'experiments').iterdir()), [])
