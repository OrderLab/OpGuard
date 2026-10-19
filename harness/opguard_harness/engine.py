"""A fixed, inspectable workflow. The agent chooses when to advance or refine."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import uuid

from .spec import validate
from .store import Store, atomic_json

STAGES = [(side, phase) for side in ('baseline', 'candidate')
          for phase in ('discover', 'prepare', 'trace')] + [('comparison', 'compare')]


def now():
    return datetime.now(timezone.utc).isoformat()


class Harness:
    def __init__(self, root):
        self.store = Store(root)

    def create(self, spec):
        spec = validate(spec)
        experiment_id = uuid.uuid4().hex
        self.store.path(experiment_id).mkdir()
        record = {'version': 1, 'id': experiment_id, 'created_at': now(), 'status': 'pending',
                  'spec': spec, 'stages': [{'side': side, 'action': action, 'status': 'pending',
                                         'attempts': []} for side, action in STAGES]}
        self.store.save(record)
        return self.status(experiment_id)

    def status(self, experiment_id):
        record = self.store.load(experiment_id)
        stage = next((s for s in record['stages'] if s['status'] != 'succeeded'), None)
        return {'id': experiment_id, 'status': record['status'], 'next':
                {'side': stage['side'], 'action': stage['action']} if stage else None,
                'stages': record['stages']}

    def results(self, experiment_id):
        record = self.store.load(experiment_id)
        last = record['stages'][-1]
        return {'id': experiment_id, 'status': record['status'],
                'comparison': last.get('result'), 'verdict': 'not_evaluated',
                'complete': record['status'] == 'succeeded'}

    def refine(self, experiment_id, spec):
        # Explicit full spec, fresh experiment; old evidence stays immutable.
        self.store.load(experiment_id)
        child = self.create(spec)
        with self.store.locked(child['id']) as record:
            record['parent'] = experiment_id
            self.store.save(record)
        return child

    def advance(self, experiment_id):
        with self.store.locked(experiment_id) as record:
            stage = next((s for s in record['stages'] if s['status'] != 'succeeded'), None)
            if stage is None:
                return self.status(experiment_id)
            if stage['status'] == 'running':
                raise RuntimeError('Previous controller interrupted. Inspect the recorded process/logs and use recover; do not launch duplicate workers')
            if len(stage['attempts']) >= record['spec']['max_stage_attempts']:
                raise RuntimeError('Stage attempt budget exhausted; refine into a new experiment')
            number = len(stage['attempts']) + 1
            attempt_dir = self.store.path(experiment_id) / f'{stage["side"]}-{stage["action"]}-{number}'
            attempt_dir.mkdir()
            request = self._request(record, stage, attempt_dir)
            atomic_json(attempt_dir / 'request.json', request)
            attempt = {'started_at': now(), 'directory': str(attempt_dir), 'status': 'running'}
            stage['attempts'].append(attempt)
            stage['status'] = record['status'] = 'running'
            self.store.save(record)
            env = {k: v for k, v in os.environ.items() if not k.startswith('OPGUARD_')}
            env['OPGUARD_AUTOSTART'] = '0'
            command = [*record['spec']['backend'], stage['action'], '--request', str(attempt_dir / 'request.json')]
            try:
                with (attempt_dir / 'response.json').open('w') as out, (attempt_dir / 'backend.log').open('w') as err:
                    # Inherit signals: Ctrl-C reaches the backend, which unwinds its
                    # source transaction. Never automatically retry interrupted jobs.
                    process = subprocess.Popen(command, stdout=out, stderr=err, env=env)
                    attempt['pid'] = process.pid
                    self.store.save(record)
                    code = process.wait()
                response = json.loads((attempt_dir / 'response.json').read_text())
                if code or response.get('protocol_version') != 1 or response.get('ok') is not True:
                    raise RuntimeError(f'Backend failed: {response.get("error", response)}')
                result = response['result']
                self._validate_result(stage['action'], request, result)
                stage['result'] = result
                stage['status'] = attempt['status'] = 'succeeded'
                record['status'] = 'succeeded' if all(s['status'] == 'succeeded' for s in record['stages']) else 'pending'
            except Exception as exc:
                stage['status'] = attempt['status'] = 'failed'
                record['status'] = 'failed'
                attempt['error'] = str(exc)
            finally:
                attempt['finished_at'] = now()
                self.store.save(record)
            return self.status(experiment_id)

    def recover(self, experiment_id):
        # A controller crash is ambiguous. Only acknowledge after backend exit and
        # completed source restoration; no stale PID guessing or automatic overwrite.
        with self.store.locked(experiment_id) as record:
            stage = next((s for s in record['stages'] if s['status'] == 'running'), None)
            if stage is None:
                raise ValueError('No interrupted running stage')
            attempt = stage['attempts'][-1]
            pid = attempt.get('pid')
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError('Recorded backend PID still exists; inspect it before recovery')
            path = Path(attempt['directory'])
            if stage['action'] == 'trace':
                state = path / 'output-instrument'
                if state.exists() and not (state / 'restored.json').exists():
                    raise RuntimeError(f'Source restoration not confirmed. Inspect originals.json in {state}; recover sources before retrying')
            # Discovery could leave detached remote workers. A successful backend
            # response proves its blocking launcher returned and receipts validated.
            response_path = path / 'response.json'
            try:
                response = json.loads(response_path.read_text())
            except (OSError, ValueError):
                raise RuntimeError('No terminal backend response; verify/cancel launcher workers before creating a new experiment')
            if response.get('protocol_version') != 1 or not isinstance(response.get('ok'), bool):
                raise RuntimeError('Invalid terminal backend response')
            if response['ok']:
                request = json.loads((path / 'request.json').read_text())
                self._validate_result(stage['action'], request, response['result'])
                stage['result'] = response['result']
                stage['status'] = attempt['status'] = 'succeeded'
                record['status'] = 'succeeded' if all(s['status'] == 'succeeded' for s in record['stages']) else 'pending'
            else:
                stage['status'] = attempt['status'] = 'failed'
                attempt['error'] = str(response.get('error'))
                record['status'] = 'failed'
            attempt['recovered_at'] = now()
            self.store.save(record)
        return self.status(experiment_id)

    @staticmethod
    def _result(record, side, action):
        return next(s['result'] for s in record['stages'] if s['side'] == side and s['action'] == action)

    def _request(self, record, stage, directory):
        spec, side, action = record['spec'], stage['side'], stage['action']
        request = {'protocol_version': 1, 'output': str(directory / 'output'),
                   'local_root': spec['local_root'], 'timeout_seconds': spec['timeout_seconds']}
        if side in ('baseline', 'candidate'):
            request['workload'] = spec[side]
        if action == 'prepare':
            request['discovery'] = self._result(record, side, 'discover')
        if action == 'trace':
            request['plan'] = self._result(record, side, 'prepare')
        if action == 'compare':
            request.update(baseline=self._result(record, 'baseline', 'trace'),
                           candidate=self._result(record, 'candidate', 'trace'),
                           ranks=spec['baseline']['ranks'])
        return request

    @staticmethod
    def _validate_result(action, request, result):
        if not isinstance(result, dict):
            raise ValueError('Backend result must be an object')
        if action in ('discover', 'trace'):
            if result.get('output') != request['output'] or result.get('ranks') != request['workload']['ranks'] or result.get('phase') != action:
                raise ValueError('Capture result does not match the requested phase/output/ranks')
        elif action == 'prepare':
            if not result.get('sources') or not Path(result['config']).is_file() or not result.get('config_sha256'):
                raise ValueError('Invalid instrumentation plan')
        else:
            if result.get('verdict') != 'not_evaluated' or result.get('ranks') != request['ranks'] or not result.get('reports'):
                raise ValueError('Invalid comparison result')
            if len(result['reports']) != len(request['ranks']) or any(not Path(p).is_file() for p in result['reports']):
                raise ValueError('Missing comparison reports')
