"""Backend operations. No agent dependency and no global CLI session state."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import uuid


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)


def clean_env():
    # No inherited tracing state in the control process or launcher.
    env = {k: v for k, v in os.environ.items() if not k.startswith('OPGUARD_') and k != 'DEFAULT_CODE_REPO_BASE'}
    env['OPGUARD_AUTOSTART'] = '0'
    return env


@contextlib.contextmanager
def source_lock(cwd):
    # All harness operations touching this checkout share this advisory lock.
    lock = Path(cwd) / '.opguard' / 'harness.lock'
    lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another harness operation owns this source checkout')
        active = lock.parent / 'harness-active.json'
        if active.exists():
            raise RuntimeError(f'Interrupted source transaction; inspect {active} before continuing')
        yield


def receipts(root, expected, phase, capture_id):
    root = Path(root)
    records = [json.loads(p.read_text()) for p in (root / '.opguard-complete').glob('rank-*.json')]
    ranks = [item['rank'] for item in records]
    if sorted(ranks) != sorted(expected):
        raise RuntimeError(f'Incomplete ranks: expected {expected}, received {sorted(ranks)}')
    from opguard.local_sync.materialize import BUNDLE_DIR
    for item in records:
        if item['capture_id'] != capture_id or item['phase'] != phase:
            raise RuntimeError('Stale or incompatible completion receipt')
        owner = item['owner']
        if Path(owner).name != owner or owner in ('.', '..'):
            raise ValueError('Invalid upload owner')
        checkpoint = item['checkpoint']
        if not checkpoint:
            raise RuntimeError(f'Rank {item["rank"]} published no trace batches')
        published = sorted((root / BUNDLE_DIR / owner).glob('*.tar'))
        # Checkpoint sequence denotes the last committed batch.
        if len(published) != checkpoint['sequence']:
            raise RuntimeError('Published batches do not match final checkpoint')
    return records


def capture(request, phase):
    workload = request['workload']
    output = Path(request['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    capture_id = uuid.uuid4().hex
    worker_request = output / 'worker.json'
    write_json(worker_request, {'output': str(output), 'phase': phase, 'capture_id': capture_id})
    env = clean_env()
    env.update(workload.get('env', {}))
    env.update({'OPGUARD_AUTOSTART': '0', 'OPGUARD_CALLSITE_TRACER': '1' if phase == 'discover' else '0',
                'OPGUARD_TRACE_OUTPUT_BASE': str(output), 'OPGUARD_LOCAL_DUMP_TARGET': str(output),
                'OPGUARD_LOCAL_DUMP_ENABLED': '1', 'OPGUARD_LOCAL_DUMP_BASE': request['local_root'],
                'OPGUARD_CALLSITE_OUTPUT_DIR': str(output / 'callsite_trace'),
                'OPGUARD_EXEC_OUTPUT_DIR': str(output / 'exec_trace'),
                'OPGUARD_RUNTIME_CALLSTACK': '1'})
    command = [*workload['launcher'], '-m', 'opguard.tools.worker', str(worker_request),
               workload['script'], *workload.get('args', [])]
    # The supplied launcher must block until all workers exit; no detached submission.
    with (output / 'workload.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=workload['cwd'], env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=request['timeout_seconds'])
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    if code != 0:
        raise RuntimeError(f'Launcher exited {code}; see {output / "workload.log"}')
    records = receipts(output, workload['ranks'], phase, capture_id)
    # Validate every archive checksum and sequence, and require nonempty rank data.
    with snapshot(output) as local:
        for rank in workload['ranks']:
            if phase == 'trace':
                candidates = (local / 'exec_trace' / f'gpu_{rank:03d}').rglob('*.jsonl')
            else:
                candidates = (local / 'callsite_trace').glob(f'*rank{rank}_*.json')
            if not any(p.stat().st_size > 2 for p in candidates):
                raise RuntimeError(f'No usable {phase} data for rank {rank}')
    return {'output': str(output), 'capture_id': capture_id, 'phase': phase,
            'ranks': workload['ranks'], 'receipts': len(records)}


@contextlib.contextmanager
def snapshot(source):
    from opguard.local_sync.materialize import has_bundles, materialize
    if has_bundles(source):
        with tempfile.TemporaryDirectory(prefix='opguard-tool-') as scratch:
            yield Path(materialize(source, Path(scratch) / 'snapshot'))
    else:
        yield Path(source).resolve()


def prepare(request):
    import yaml
    from opguard.preprocessing import PreprocessingPipeline
    output = Path(request['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cwd = Path(request['workload']['cwd']).resolve()
    with snapshot(request['discovery']['output']) as local:
        summary = PreprocessingPipeline(base_dir=str(cwd)).run_pipeline(
            inputs=[str(local / 'callsite_trace')], output_dir=str(output))
    selected = summary.get('sync_yaml') or summary.get('filtered_yaml') or summary.get('final_combined_yaml')
    if not selected or not Path(selected).is_file():
        raise RuntimeError('Preprocessing produced no instrumentation config')
    config = yaml.safe_load(Path(selected).read_text())
    # Restrict v1 to in-checkout callsite instrumentation. Global/module rewriting
    # needs a separate transactional implementation before being automated.
    if config.get('meta_vars') or config.get('entry'):
        raise ValueError('Harness v1 does not support global meta_vars/entry injections')
    entries = config.get('instrumentations', [])
    kept, skipped = [], []
    fingerprints = {}
    for item in entries:
        path = Path(item['file_path'])
        path = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
        if cwd not in path.parents:
            skipped.append(str(path))
            continue
        if not path.is_file():
            raise ValueError(f'Instrumentation source missing: {path}')
        kept.append({**item, 'file_path': str(path)})
        fingerprints[str(path)] = digest(path)
    if not kept:
        raise RuntimeError('No instrumentable callsites within the workload checkout')
    normalized = output / 'capture.yaml'
    normalized.write_text(yaml.safe_dump({'instrumentations': kept, 'meta_vars': [], 'entry': []}))
    plan = {'config': str(normalized), 'config_sha256': digest(normalized),
            'sources': fingerprints, 'excluded_files': sorted(set(skipped)),
            'coverage': 'observed_callsites_only'}
    write_json(output / 'plan.json', plan)
    return plan


def remove_bytecode(source):
    source = Path(source)
    for cached in (source.parent / '__pycache__').glob(source.stem + '.*.pyc'):
        cached.unlink()


def trace(request):
    import yaml
    from opguard import instrument_all_files as instrument
    plan = request['plan']
    cwd = Path(request['workload']['cwd']).resolve()
    if digest(plan['config']) != plan['config_sha256']:
        raise ValueError('Instrumentation plan changed; prepare again')
    for path, expected in plan['sources'].items():
        if cwd not in Path(path).resolve().parents or digest(path) != expected:
            raise ValueError(f'Source changed or outside checkout: {path}')
    config = yaml.safe_load(Path(plan['config']).read_text())
    configured = {str(Path(item['file_path']).resolve()) for item in config.get('instrumentations', [])}
    if configured != set(plan['sources']) or config.get('meta_vars') or config.get('entry'):
        raise ValueError('Plan source manifest does not match its instrumentation config')
    # Private manifest per operation, never the interactive CLI's manifest.
    state = Path(request['output']).resolve().with_name(Path(request['output']).name + '-instrument')
    state.mkdir(parents=True, exist_ok=False)
    instrument.OPGUARD_DIR = state
    instrument.BACKUP_ROOT = state / 'backups'
    instrument.MANIFEST = state / 'manifest.json'
    os.environ['DEFAULT_CODE_REPO_BASE'] = str(cwd)
    os.environ['OPGUARD_CONFIG_PATH'] = plan['config']
    os.environ['OPGUARD_CALLSITE_CONFIG'] = plan['config']
    os.environ['OPGUARD_TRACE_OUTPUT_BASE'] = str(state / 'artifacts')
    # Durable source copies also cover an interrupted instrumenter before manifest save.
    originals = {}
    for index, path in enumerate(plan['sources']):
        backup = state / f'source-{index}.py'
        backup.write_bytes(Path(path).read_bytes())
        originals[path] = str(backup)
    write_json(state / 'originals.json', originals)
    active = cwd / '.opguard' / 'harness-active.json'
    write_json(active, {'state': str(state)})
    instrumented_hashes = None
    try:
        if not instrument.run_instrumentation(plan['config']):
            raise RuntimeError('Instrumentation failed')
        instrumented_hashes = {path: digest(path) for path in originals}
        for path in originals:
            remove_bytecode(path)
        return capture(request, 'trace')
    finally:
        # This checkout is exclusively owned for the whole operation. Workloads
        # must not edit their source files. Originals survive even abrupt termination.
        changed = []
        for path, backup in originals.items():
            if instrumented_hashes is not None and (Path(path).is_symlink() or not Path(path).is_file() or digest(path) != instrumented_hashes[path]):
                changed.append(path)
                continue
            Path(path).write_bytes(Path(backup).read_bytes())
            remove_bytecode(path)
        if changed:
            raise RuntimeError(f'Sources edited during capture; retained edits and backups: {changed}')
        write_json(state / 'restored.json', {'restored': True})
        active.unlink()


def compare(request):
    from opguard.trace_export.perfetto.build import build_perfetto
    output = Path(request['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    ranks = request['ranks']
    reports = []
    with snapshot(request['baseline']['output']) as left, snapshot(request['candidate']['output']) as right:
        for rank in ranks:
            a, b = (root / 'exec_trace' / f'gpu_{rank:03d}' for root in (left, right))
            if not a.is_dir() or not b.is_dir():
                raise RuntimeError(f'Missing comparison rank {rank}')
            target = output / f'perfetto_gpu_{rank:03d}.json.gz'
            build_perfetto(str(a), str(b), str(target), ignore_fields={})
            if not target.is_file() or not target.stat().st_size:
                raise RuntimeError(f'No comparison report for rank {rank}')
            reports.append(str(target))
    # Perfetto generation is not a numerical equivalence verdict.
    return {'reports': reports, 'ranks': ranks, 'verdict': 'not_evaluated',
            'coverage': 'observed_callsites_only'}


def dispatch(action, request):
    if action == 'align':
        from opguard.trace_export.perfetto.alignment_map import render_alignment
        return render_alignment(request['artifact'], request['alignment_map'], request['output'])
    if action in ('discover', 'prepare', 'trace'):
        cwd = request['workload']['cwd']
        with source_lock(cwd):
            if action == 'discover':
                return capture(request, 'discover')
            if action == 'prepare':
                return prepare(request)
            return trace(request)
    if action == 'compare':
        return compare(request)
    raise ValueError(f'Unknown action: {action}')
