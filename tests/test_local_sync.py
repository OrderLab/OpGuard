import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import random

import pytest

from opguard import distributed as identity
from opguard.local_sync import bundles, transport
from opguard.local_sync.manager import LocalDumpManager
from opguard.local_sync.materialize import materialize, BUNDLE_DIR
from opguard.local_sync.paths import ShadowPaths


@pytest.fixture
def manager(tmp_path, monkeypatch):
    for key in identity.GLOBAL_RANK_ENV + identity.LOCAL_RANK_ENV + identity.WORLD_SIZE_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delitem(sys.modules, 'torch.distributed', raising=False)
    monkeypatch.setenv('OPGUARD_LOCAL_DUMP_BASE', str(tmp_path / 'local'))
    monkeypatch.setenv('OPGUARD_SYNC_MBPS', '100000')
    return LocalDumpManager(tmp_path / 'shared')


def write(manager, name, content):
    path = Path(manager.local_base) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def archives(manager):
    return sorted(Path(manager.target_dir).rglob('*.tar'))


def restore(manager, tmp_path, name='restored'):
    return Path(materialize(manager.target_dir, tmp_path / name))


def test_periodic_upload_does_not_consume_workload_random_state(manager):
    before = random.getstate()
    manager.start_periodic_sync()
    assert manager.stop_periodic_sync()
    assert random.getstate() == before


def test_delta_upload_and_identical_sync_is_noop(manager, tmp_path):
    log = write(manager, 'exec_trace/gpu_000/ops/test.jsonl', b'{"x":1}\n')
    write(manager, 'exec_trace/gpu_000/ops/full/test.pt', b'tensor')
    manager.sync_all()
    first = archives(manager)
    manager.sync_all()
    assert archives(manager) == first
    with log.open('ab') as f:
        f.write(b'{"x":2}\nunfinished')
    manager.sync_all()
    manifest = bundles.read_manifest(archives(manager)[-1])
    assert len(manifest['entries']) == 1
    assert manifest['entries'][0]['mode'] == 'append'
    assert manifest['entries'][0]['offset'] == len(b'{"x":1}\n')
    assert manifest['entries'][0]['size'] == len(b'{"x":2}\n')
    result = restore(manager, tmp_path)
    assert (result / 'exec_trace/gpu_000/ops/test.jsonl').read_bytes() == b'{"x":1}\n{"x":2}\n'
    assert (result / 'exec_trace/gpu_000/ops/full/test.pt').read_bytes() == b'tensor'


def test_bounded_batches_and_complete_lines(manager, tmp_path):
    manager.max_bytes = 12
    data = b'12345\n' * 10
    write(manager, 'exec_trace/gpu_000/log.jsonl', data)
    manager.sync_all()
    assert len(archives(manager)) == 5
    for file in archives(manager):
        assert sum(e['size'] for e in bundles.read_manifest(file)['entries']) <= 12
    assert (restore(manager, tmp_path) / 'exec_trace/gpu_000/log.jsonl').read_bytes() == data


def test_periodic_uploads_only_one_batch(manager):
    manager.max_bytes = 12
    write(manager, 'exec_trace/gpu_000/log.jsonl', b'12345\n' * 10)
    manager.sync_all(force=False)
    assert len(archives(manager)) == 1


@pytest.mark.parametrize('initial_clock', [0.0, 10.0, 1000000.0])
def test_periodic_interval_starts_after_first_upload(manager, monkeypatch, initial_clock):
    clock = [initial_clock]
    monkeypatch.setattr('opguard.local_sync.manager.time.monotonic', lambda: clock[0])
    manager.max_bytes = 12
    write(manager, 'exec_trace/gpu_000/log.jsonl', b'12345\n' * 10)
    manager.sync_all(force=False)
    assert len(archives(manager)) == 1
    clock[0] += manager.sync_interval_sec - 1
    manager.sync_all(force=False)
    assert len(archives(manager)) == 1
    clock[0] += 1
    manager.sync_all(force=False)
    assert len(archives(manager)) == 2


def test_incomplete_record_waits_for_newline(manager, tmp_path):
    log = write(manager, 'exec_trace/gpu_000/log.jsonl', b'partial')
    manager.sync_all()
    assert not archives(manager)
    with log.open('ab') as f:
        f.write(b'\n')
    manager.sync_all()
    assert (restore(manager, tmp_path) / 'exec_trace/gpu_000/log.jsonl').read_bytes() == b'partial\n'


def test_oversized_record_fails_explicitly(manager):
    manager.max_bytes = 4
    write(manager, 'exec_trace/gpu_000/log.jsonl', b'0123456789\n')
    with pytest.raises(ValueError, match='batch limit'):
        manager.sync_all()


def test_retry_after_upload_failure_keeps_cursor_and_payload(manager, monkeypatch, tmp_path):
    log = write(manager, 'exec_trace/gpu_000/log.jsonl', b'first\n')
    real_publish = transport.publish
    monkeypatch.setattr(transport, 'publish', lambda *a, **kw: (_ for _ in ()).throw(OSError('offline')))
    with pytest.raises(OSError, match='offline'):
        manager.sync_all()
    _, _, spool = manager._identity()
    assert (spool / 'pending.tar').is_file()
    assert bundles.load_state(spool)['sequence'] == 0
    with log.open('ab') as f:
        f.write(b'second\n')
    monkeypatch.setattr(transport, 'publish', real_publish)
    restarted = LocalDumpManager(manager.target_dir)
    restarted.sync_all()
    result = restore(restarted, tmp_path)
    assert (result / 'exec_trace/gpu_000/log.jsonl').read_bytes() == b'first\nsecond\n'
    assert not (spool / 'pending.tar').exists()


def test_crash_after_publish_before_checkpoint_is_idempotent(manager, monkeypatch, tmp_path):
    write(manager, 'exec_trace/gpu_000/log.jsonl', b'once\n')
    real_commit = bundles.commit
    monkeypatch.setattr(bundles, 'commit', lambda *a: (_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(OSError):
        manager.sync_all()
    first = archives(manager)
    monkeypatch.setattr(bundles, 'commit', real_commit)
    manager.sync_all()
    assert archives(manager) == first
    assert (restore(manager, tmp_path) / 'exec_trace/gpu_000/log.jsonl').read_bytes() == b'once\n'


@pytest.mark.parametrize('replacement', [b'new\n', b''])
def test_truncated_log_replaces_old_snapshot(manager, tmp_path, replacement):
    log = write(manager, 'exec_trace/gpu_000/log.jsonl', b'old long line\n')
    manager.sync_all()
    log.write_bytes(replacement)
    manager.sync_all()
    assert (restore(manager, tmp_path) / 'exec_trace/gpu_000/log.jsonl').read_bytes() == replacement


def test_atomic_replacement_of_nonlog_file(manager, tmp_path):
    file = write(manager, 'exec_trace/gpu_000/tensor.pt', b'old')
    manager.sync_all()
    temp = file.with_suffix('.tmp')
    temp.write_bytes(b'new')
    os.replace(temp, file)
    manager.sync_all()
    assert (restore(manager, tmp_path) / 'exec_trace/gpu_000/tensor.pt').read_bytes() == b'new'


def test_ownership_and_temp_files(manager):
    own = write(manager, 'exec_trace/gpu_000/log.jsonl', b'own\n')
    write(manager, 'exec_trace/gpu_001/log.jsonl', b'other\n')
    write(manager, 'exec_trace/gpu_000/partial.tmp', b'incomplete')
    call = write(manager, 'callsite_trace/python_rank0_pid1.json', b'{}')
    write(manager, 'callsite_trace/python_rank1_pid2.json', b'{}')
    assert set(manager.paths.owned_files(0, 0)) == {own, call}


def test_mapping_honors_custom_root_and_rejects_prefix_collision(manager, tmp_path):
    assert Path(manager.local_base).is_relative_to(tmp_path / 'local')
    local = manager.get_local_path(manager.target_dir + '/exec_trace')
    assert manager.get_local_path(local) == local
    with pytest.raises(ValueError, match='outside'):
        manager.get_local_path(manager.target_dir + '-other/file')


def test_atomic_publish_hides_partial_and_cancellation_retains_local(tmp_path):
    source = tmp_path / 'bundle.tar'
    source.write_bytes(b'x' * 2048)
    target = tmp_path / 'shared/final.tar'
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        transport.publish(source, target, tmp_path / 'lock', 1, stop)
    assert source.exists() and not target.exists()
    assert not list(target.parent.glob('*.part'))


def test_corruption_never_publishes_restore(manager, tmp_path):
    write(manager, 'exec_trace/gpu_000/log.jsonl', b'valid\n')
    manager.sync_all()
    archives(manager)[0].write_bytes(b'bad')
    with pytest.raises(ValueError, match='checksum'):
        restore(manager, tmp_path)
    assert not (tmp_path / 'restored').exists()


def test_missing_batch_is_detected(manager, tmp_path):
    manager.max_bytes = 4
    write(manager, 'exec_trace/gpu_000/log.jsonl', b'abc\n' * 2)
    manager.sync_all()
    archives(manager)[0].unlink()
    with pytest.raises(ValueError, match='missing'):
        restore(manager, tmp_path)


def test_callsite_only_job_is_uploaded(manager, tmp_path):
    write(manager, 'callsite_trace/python_rank0_pid1.json', b'{"site":1}')
    manager.sync_all()
    assert (restore(manager, tmp_path) / 'callsite_trace/python_rank0_pid1.json').read_bytes() == b'{"site":1}'


def test_different_processes_same_owner_commit_once(manager, tmp_path):
    write(manager, 'exec_trace/gpu_000/log.jsonl', b'once\n')
    code = 'from opguard.local_dump_manager import LocalDumpManager; import sys; LocalDumpManager(sys.argv[1]).sync_all()'
    env = dict(os.environ)
    env.update(RANK='0', LOCAL_RANK='0', WORLD_SIZE='2')
    processes = [subprocess.Popen([sys.executable, '-c', code, manager.target_dir], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    for process in processes:
        _, err = process.communicate(timeout=20)
        assert process.returncode == 0, err
    assert len(archives(manager)) == 1
    assert (restore(manager, tmp_path) / 'exec_trace/gpu_000/log.jsonl').read_bytes() == b'once\n'


def test_node_metadata_is_uploaded_once_and_restored_per_rank(manager, tmp_path, monkeypatch):
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    write(manager, 'exec_trace/source_files/file_index.json', b'{"source":"node-a"}')
    write(manager, 'exec_trace/gpu_000/log.jsonl', b'a\n')
    manager.sync_all()
    monkeypatch.setenv('OPGUARD_NODE_ID', 'second-node')
    monkeypatch.setenv('OPGUARD_LOCAL_DUMP_BASE', str(tmp_path / 'other-local'))
    monkeypatch.setenv('RANK', '2')
    other = LocalDumpManager(manager.target_dir)
    write(other, 'exec_trace/source_files/file_index.json', b'{"source":"node-b"}')
    write(other, 'exec_trace/gpu_002/log.jsonl', b'b\n')
    other.sync_all()
    result = restore(manager, tmp_path)
    assert json.loads((result / 'exec_trace/gpu_000/source_files/file_index.json').read_text()) == {'source': 'node-a'}
    assert json.loads((result / 'exec_trace/gpu_002/source_files/file_index.json').read_text()) == {'source': 'node-b'}


def test_atomic_producer_only_exposes_complete_file(manager):
    from opguard.local_sync.paths import atomic_output
    target = Path(manager.local_base) / 'exec_trace/gpu_000/tensor.pt'
    with atomic_output(target) as temporary:
        Path(temporary).write_bytes(b'partial')
        assert not target.exists()
        assert not list(manager.paths.owned_files(0, 0))
        Path(temporary).write_bytes(b'complete')
    assert target.read_bytes() == b'complete'


def test_rate_limiter_and_publish_visibility(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.write_bytes(b'x' * 100)
    destination = tmp_path / 'remote/result.tar'
    delays = []
    monkeypatch.setattr(transport.time, 'monotonic', lambda: 0)
    def wait(delay):
        assert not destination.exists()
        delays.append(delay)
    monkeypatch.setattr(transport.time, 'sleep', wait)
    transport.publish(source, destination, tmp_path / 'node.lock', 50)
    assert delays == [2]
    assert destination.read_bytes() == source.read_bytes()


def test_missing_staging_target_never_writes_shared(monkeypatch):
    from opguard import local_dump_manager
    monkeypatch.setattr(local_dump_manager, '_local_dump_manager', None)
    monkeypatch.setenv('OPGUARD_LOCAL_DUMP_ENABLED', '1')
    monkeypatch.delenv('OPGUARD_LOCAL_DUMP_TARGET', raising=False)
    with pytest.raises(ValueError, match='requires'):
        local_dump_manager.translate_path('/shared/run')


def test_lost_checkpoint_fails_instead_of_corrupting_history(manager):
    write(manager, 'exec_trace/gpu_000/log.jsonl', b'first\n')
    manager.sync_all()
    _, _, spool = manager._identity()
    (spool / 'checkpoint.json').unlink()
    with pytest.raises(RuntimeError, match='checkpoint is missing'):
        manager.sync_all()


def test_cli_compare_restores_bundles_locally(manager, tmp_path, monkeypatch):
    from argparse import Namespace
    from opguard.cli import OpGuardCLI
    write(manager, 'exec_trace/gpu_000/log.jsonl', b'first\n')
    manager.sync_all()
    cli = OpGuardCLI.__new__(OpGuardCLI)
    captured = []
    def compare(args):
        assert (Path(args.dir1) / 'exec_trace/gpu_000/log.jsonl').read_bytes() == b'first\n'
        assert args._report_dir1 == manager.target_dir
        assert args.dir1 != manager.target_dir
        captured.append(args.dir1)
    monkeypatch.setattr(cli, '_compare_local', compare)
    cli.compare(Namespace(dir1=manager.target_dir, dir2=manager.target_dir))
    assert len(captured) == 1
    assert not Path(captured[0]).exists()


def test_target_selection_does_not_start_staging(tmp_path, monkeypatch):
    from opguard import _running
    monkeypatch.setenv('OPGUARD_TRACE_OUTPUT_BASE', str(tmp_path / 'shared'))
    monkeypatch.setattr('opguard.local_dump_manager.translate_path', lambda _: pytest.fail('must not translate target'))
    assert _running.get_trace_target_dir() == str(tmp_path / 'shared')


def test_rank_ownership_survives_process_group_teardown(manager, monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'torch.distributed', SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: True,
        get_rank=lambda: 8, get_world_size=lambda: 16))
    monkeypatch.setenv('LOCAL_RANK', '0')
    manager._identity()
    monkeypatch.delitem(sys.modules, 'torch.distributed')
    monkeypatch.delenv('LOCAL_RANK')
    write(manager, 'exec_trace/gpu_008/log.jsonl', b'last\n')
    manager.sync_all()
    assert (restore(manager, tmp_path) / 'exec_trace/gpu_008/log.jsonl').read_bytes() == b'last\n'


def test_record_that_does_not_fit_remainder_moves_to_next_batch(manager, tmp_path):
    manager.max_bytes = 8
    write(manager, 'exec_trace/gpu_000/a.pt', b'12345')
    write(manager, 'exec_trace/gpu_000/b.jsonl', b'1234567\n')
    manager.sync_all()
    assert len(archives(manager)) == 2
    assert (restore(manager, tmp_path) / 'exec_trace/gpu_000/b.jsonl').read_bytes() == b'1234567\n'


def test_node_lock_serializes_different_processes(tmp_path):
    code = '''
import sys, time, json
from opguard.local_sync.transport import local_lock
with local_lock(sys.argv[1]):
    start = time.monotonic()
    time.sleep(0.05)
    print(json.dumps([start, time.monotonic()]))
'''
    processes = [subprocess.Popen([sys.executable, '-c', code, str(tmp_path / 'node.lock')], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(3)]
    times = []
    for process in processes:
        out, err = process.communicate(timeout=10)
        assert process.returncode == 0, err
        times.append(json.loads(out))
    times.sort()
    assert all(a[1] <= b[0] for a, b in zip(times, times[1:]))


def test_restore_rejects_path_traversal(manager, tmp_path):
    import io
    import tarfile
    archive = tmp_path / 'malicious.tar'
    manifest = {'version': 1, 'sequence': 1, 'entries': [{'path': '../escaped'}]}
    with tarfile.open(archive, 'w') as tar:
        data = json.dumps(manifest).encode()
        info = tarfile.TarInfo('manifest.json')
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    owner = Path(manager.target_dir) / BUNDLE_DIR / 'owner'
    owner.mkdir(parents=True)
    archive.rename(owner / f'000000000001-{bundles.digest(archive)}.tar')
    with pytest.raises(ValueError, match='Unsafe'):
        restore(manager, tmp_path)
    assert not (tmp_path / 'escaped').exists()


def test_failed_atomic_producer_keeps_previous_file(manager):
    from opguard.local_sync.paths import atomic_output
    file = write(manager, 'exec_trace/gpu_000/tensor.pt', b'old')
    with pytest.raises(OSError):
        with atomic_output(file) as temp:
            Path(temp).write_bytes(b'half')
            raise OSError('save failed')
    assert file.read_bytes() == b'old'
    assert not list(file.parent.glob('*.part'))
