import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from opguard.tools.operations import digest, receipts, source_lock, trace, write_json


def test_capabilities_without_tracing_dependencies():
    result = subprocess.run([sys.executable, '-m', 'opguard.tools', 'capabilities'], capture_output=True, text=True)
    assert result.returncode == 0
    response = json.loads(result.stdout)
    assert response['ok'] and not response['result']['numeric_verdict']


def test_invalid_protocol_is_json_error(tmp_path):
    request = tmp_path / 'request.json'
    request.write_text('{"protocol_version":999}')
    result = subprocess.run([sys.executable, '-m', 'opguard.tools', 'discover', '--request', str(request)], capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)['error']['type'] == 'ValueError'


def test_receipts_require_all_ranks_and_batches(tmp_path):
    with pytest.raises(RuntimeError, match='Incomplete ranks'):
        receipts(tmp_path, [0, 1], 'trace', 'capture')
    record = dict(rank=0, capture_id='capture', phase='trace', owner='node-rank000000', checkpoint={'sequence': 1})
    write_json(tmp_path / '.opguard-complete' / 'rank-0.json', record)
    with pytest.raises(RuntimeError, match='batches'):
        receipts(tmp_path, [0], 'trace', 'capture')
    batch = tmp_path / '.opguard-bundles' / record['owner'] / '000000000001-test.tar'
    batch.parent.mkdir(parents=True)
    batch.write_bytes(b'placeholder')
    assert len(receipts(tmp_path, [0], 'trace', 'capture')) == 1
    with pytest.raises(RuntimeError, match='Stale'):
        receipts(tmp_path, [0], 'trace', 'another')


def test_source_lock_and_interrupted_transaction(tmp_path):
    with source_lock(tmp_path):
        with pytest.raises(RuntimeError, match='owns'):
            with source_lock(tmp_path):
                pass
    write_json(tmp_path / '.opguard' / 'harness-active.json', {'state': 'interrupted'})
    with pytest.raises(RuntimeError, match='Interrupted'):
        with source_lock(tmp_path):
            pass


def trace_request(tmp_path):
    import yaml
    cwd = tmp_path / 'source'
    cwd.mkdir()
    (cwd / '.opguard').mkdir()
    source = cwd / 'train.py'
    source.write_text('original = True\n')
    config = tmp_path / 'plan.yaml'
    config.write_text(yaml.safe_dump({'instrumentations': [{'file_path': str(source)}]}))
    return {'output': str(tmp_path / 'trace'), 'workload': {'cwd': str(cwd)},
            'plan': {'config': str(config), 'config_sha256': digest(config), 'sources': {str(source): digest(source)}}}, source


def test_trace_restores_after_capture_failure(tmp_path, monkeypatch):
    from opguard import instrument_all_files as instrument
    request, source = trace_request(tmp_path)
    # isolate all process-local module/env changes made by this process-only API
    for attr in ('OPGUARD_DIR', 'BACKUP_ROOT', 'MANIFEST'):
        monkeypatch.setattr(instrument, attr, getattr(instrument, attr))
    for name in ('DEFAULT_CODE_REPO_BASE', 'OPGUARD_CONFIG_PATH', 'OPGUARD_CALLSITE_CONFIG', 'OPGUARD_TRACE_OUTPUT_BASE'):
        monkeypatch.setenv(name, '')
    def inject(*args):
        source.write_text('instrumented = True\n')
        return True
    monkeypatch.setattr(instrument, 'run_instrumentation', inject)
    with patch('opguard.tools.operations.capture', side_effect=RuntimeError('capture failed')):
        with pytest.raises(RuntimeError, match='capture failed'):
            trace(request)
    assert source.read_text() == 'original = True\n'
    assert (tmp_path / 'trace-instrument' / 'restored.json').exists()
    assert not (source.parent / '.opguard' / 'harness-active.json').exists()


def test_trace_preserves_edits_during_workload(tmp_path, monkeypatch):
    from opguard import instrument_all_files as instrument
    request, source = trace_request(tmp_path)
    for attr in ('OPGUARD_DIR', 'BACKUP_ROOT', 'MANIFEST'):
        monkeypatch.setattr(instrument, attr, getattr(instrument, attr))
    for name in ('DEFAULT_CODE_REPO_BASE', 'OPGUARD_CONFIG_PATH', 'OPGUARD_CALLSITE_CONFIG', 'OPGUARD_TRACE_OUTPUT_BASE'):
        monkeypatch.setenv(name, '')
    monkeypatch.setattr(instrument, 'run_instrumentation', lambda *args: True)
    def edited(*args):
        source.write_text('user_edit = True\n')
        return {}
    with patch('opguard.tools.operations.capture', edited):
        with pytest.raises(RuntimeError, match='Sources edited'):
            trace(request)
    assert source.read_text() == 'user_edit = True\n'
    assert (source.parent / '.opguard' / 'harness-active.json').exists()


def test_plan_changes_fail_before_instrumentation(tmp_path):
    request, source = trace_request(tmp_path)
    source.write_text('changed = True\n')
    with pytest.raises(ValueError, match='Source changed'):
        trace(request)
    assert not (tmp_path / 'trace-instrument').exists()


def capture_request(tmp_path, source, ranks=None, launcher=None):
    script = tmp_path / 'workload.py'
    script.write_text(source)
    return {'output': str(tmp_path / 'output'), 'local_root': str(tmp_path / 'ssd'),
            'timeout_seconds': 5, 'workload': {
                'cwd': str(tmp_path), 'script': str(script), 'args': [],
                'launcher': launcher or [sys.executable], 'ranks': ranks or [0],
                'env': {'PYTHONPATH': str(Path(__file__).resolve().parents[1])}}}


TRACE_WORKLOAD = '''
from pathlib import Path
from opguard.distributed import get_global_rank
from opguard.local_dump_manager import translate_path
import os
root = Path(translate_path(os.environ['OPGUARD_TRACE_OUTPUT_BASE']))
path = root / 'exec_trace' / ('gpu_%03d' % get_global_rank()) / 'trace.jsonl'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text('{"event":"test"}\\n')
'''


def test_real_worker_upload_and_completion(tmp_path):
    from opguard.tools.operations import capture, snapshot
    request = capture_request(tmp_path, TRACE_WORKLOAD)
    result = capture(request, 'trace')
    assert result['ranks'] == [0]
    assert result['receipts'] == 1
    with snapshot(result['output']) as local:
        assert (local / 'exec_trace' / 'gpu_000' / 'trace.jsonl').read_text() == '{"event":"test"}\n'


def test_two_real_rank_workers(tmp_path):
    from opguard.tools.operations import capture
    launcher = tmp_path / 'launcher.py'
    launcher.write_text('''
import os, subprocess, sys
processes = []
for rank in range(2):
    env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', LOCAL_WORLD_SIZE='2')
    processes.append(subprocess.Popen([sys.executable, *sys.argv[1:]], env=env))
sys.exit(max(p.wait() for p in processes))
''')
    request = capture_request(tmp_path, TRACE_WORKLOAD, [0, 1], [sys.executable, str(launcher)])
    assert capture(request, 'trace')['receipts'] == 2


def test_successful_process_without_trace_is_not_successful_capture(tmp_path):
    from opguard.tools.operations import capture
    request = capture_request(tmp_path, 'print("no trace")')
    with pytest.raises(RuntimeError, match='no trace batches'):
        capture(request, 'trace')


def test_failed_workload_has_no_completion_receipt(tmp_path):
    from opguard.tools.operations import capture
    request = capture_request(tmp_path, TRACE_WORKLOAD + '\nraise RuntimeError("failed")\n')
    with pytest.raises(RuntimeError, match='Launcher exited'):
        capture(request, 'trace')
    assert not list((tmp_path / 'output' / '.opguard-complete').glob('*.json'))


def test_workload_timeout_is_failure(tmp_path):
    from opguard.tools.operations import capture
    request = capture_request(tmp_path, 'import time; time.sleep(60)')
    request['timeout_seconds'] = 0.1
    with pytest.raises(subprocess.TimeoutExpired):
        capture(request, 'trace')


def test_real_preprocessing_protocol(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    script = source / 'model.py'
    script.write_text('def forward(x):\n    return abs(x)\n')
    discovery = tmp_path / 'discovery'
    calls = discovery / 'callsite_trace'
    calls.mkdir(parents=True)
    callsite = f'{script}:2 in forward'
    write_json(calls / 'python_callsite_rank0_pid1.json', {
        callsite: {'count': 1, 'callstacks': [[[str(script), 2, 'forward']]], 'ops': ['abs']}})
    request = tmp_path / 'request.json'
    write_json(request, {'protocol_version': 1, 'output': str(tmp_path / 'prepared'),
                         'workload': {'cwd': str(source)}, 'discovery': {'output': str(discovery)}})
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]), OPGUARD_AUTOSTART='0')
    result = subprocess.run([sys.executable, '-m', 'opguard.tools', 'prepare', '--request', str(request)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response['ok']
    assert response['result']['sources'][str(script)] == digest(script)
