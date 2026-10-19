"""Portable and deterministic paths across simulated node/rank processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import opguard._running as running
from opguard import distributed as identity


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(running, '_CFG_PATH', str(tmp_path / 'running.yaml'))
    monkeypatch.setattr(running, 'TRACE_BASE', str(tmp_path / 'traces'))
    monkeypatch.setattr(running, '_trace_output_cache', {})
    for key in identity.GLOBAL_RANK_ENV + identity.LOCAL_RANK_ENV + identity.WORLD_SIZE_ENV + (
        'OPGUARD_TRACE_OUTPUT_BASE', 'OPGUARD_TRACE_ROOT', 'OPGUARD_RUN_ID', 'OPGUARD_RUN_TOKEN',
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delitem(sys.modules, 'torch.distributed', raising=False)
    monkeypatch.setenv('OPGUARD_TRACE_OUTDIR', '')
    monkeypatch.setattr('opguard.local_dump_manager.translate_path', lambda path: path)


def test_default_shared_by_tags_without_writing_config():
    output = running.get_trace_output_dir('callsites')
    assert Path(output).is_dir()
    assert Path(output).parent == Path(running.TRACE_BASE)
    assert running.get_trace_output_dir('runtime') == output
    assert not Path(running._CFG_PATH).exists()
    assert not (Path(running.TRACE_BASE) / '.trace_dir_map').exists()


@pytest.mark.parametrize('relative', [False, True])
def test_explicit_output(tmp_path, monkeypatch, relative):
    value = 'my-run' if relative else str(tmp_path / 'explicit')
    monkeypatch.setenv('OPGUARD_TRACE_OUTPUT_BASE', value)
    expected = str(Path(running.TRACE_BASE) / value) if relative else value
    assert running.get_trace_output_dir() == expected
    assert os.environ['OPGUARD_TRACE_OUTDIR'] == expected


def test_output_precedence_and_configuration_changes(tmp_path, monkeypatch):
    saved = str(tmp_path / 'saved')
    (tmp_path / '.opguard_state.yaml').write_text(yaml.safe_dump({'output_dir': saved}))
    assert running.get_trace_output_dir() == saved
    monkeypatch.setenv('OPGUARD_RUN_TOKEN', 'legacy')
    assert running.get_trace_output_dir().endswith('/run_legacy')
    monkeypatch.setenv('OPGUARD_RUN_ID', 'new')
    assert running.get_trace_output_dir().endswith('/run_new')
    explicit = str(tmp_path / 'explicit')
    monkeypatch.setenv('OPGUARD_TRACE_OUTPUT_BASE', explicit)
    assert running.get_trace_output_dir() == explicit


@pytest.mark.parametrize('value', ['..', '.', '../escape', '/absolute', 'nested/path', 'windows\\path'])
def test_run_id_cannot_escape_root(monkeypatch, value):
    monkeypatch.setenv('OPGUARD_RUN_ID', value)
    with pytest.raises(ValueError, match='directory name'):
        running.get_trace_output_dir()


def test_template_rejected_for_distributed_launch(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '16')
    monkeypatch.setenv('RANK', '0')
    with pytest.raises(ValueError, match='Distributed tracing'):
        running.get_trace_output_dir()


def test_literal_config_is_reused_and_not_rewritten(tmp_path):
    target = tmp_path / 'existing'
    target.mkdir()
    config = yaml.safe_dump({'trace': {'output_base': str(target)}})
    Path(running._CFG_PATH).write_text(config)
    assert running.get_trace_output_dir() == str(target)
    assert Path(running._CFG_PATH).read_text() == config


def test_translation_is_not_cached(monkeypatch, tmp_path):
    monkeypatch.setenv('OPGUARD_RUN_ID', 'job')
    first = running.get_trace_output_dir()
    shadow = str(tmp_path / 'shadow')
    monkeypatch.setattr('opguard.local_dump_manager.translate_path', lambda path: shadow)
    assert running.get_trace_output_dir() == shadow
    monkeypatch.setattr('opguard.local_dump_manager.translate_path', lambda path: path)
    assert running.get_trace_output_dir() == first


def test_unwritable_output_does_not_fall_back(monkeypatch, tmp_path):
    file = tmp_path / 'file'
    file.write_text('not a directory')
    monkeypatch.setenv('OPGUARD_TRACE_OUTPUT_BASE', str(file / 'child'))
    with pytest.raises(OSError):
        running.get_trace_output_dir()


@pytest.mark.parametrize('mode', ['run_id', 'explicit'])
def test_two_nodes_four_ranks_have_one_root_and_distinct_rank_files(tmp_path, mode):
    code = '''
import json
from pathlib import Path
from opguard.distributed import get_global_rank, get_local_rank
from opguard._running import get_trace_output_dir
root = get_trace_output_dir()
rank = get_global_rank()
folder = Path(root) / 'exec_trace' / f'gpu_{rank:03d}'
folder.mkdir(parents=True, exist_ok=True)
(folder / 'owner').write_text(str(rank))
print(json.dumps([root, rank, get_local_rank()]))
'''
    processes = []
    for rank in range(4):
        env = {k: v for k, v in os.environ.items() if not k.startswith('OPGUARD_') and k not in identity.GLOBAL_RANK_ENV + identity.LOCAL_RANK_ENV + identity.WORLD_SIZE_ENV}
        env.update(SLURM_PROCID=str(rank), SLURM_LOCALID=str(rank % 2), WORLD_SIZE='4')
        env['OPGUARD_TRACE_ROOT'] = str(tmp_path)
        if mode == 'run_id':
            env['OPGUARD_RUN_ID'] = 'distributed-test'
        else:
            env['OPGUARD_TRACE_OUTPUT_BASE'] = str(tmp_path / 'explicit')
        processes.append(subprocess.Popen([sys.executable, '-c', code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    results = []
    for process in processes:
        out, err = process.communicate(timeout=20)
        assert process.returncode == 0, err
        results.append(json.loads(out.strip().splitlines()[-1]))
    assert len({r[0] for r in results}) == 1
    assert [r[1:] for r in results] == [[0, 0], [1, 1], [2, 0], [3, 1]]
    root = Path(results[0][0])
    assert sorted(p.read_text() for p in root.glob('exec_trace/gpu_*/owner')) == ['0', '1', '2', '3']
    assert not (tmp_path / '.trace_dir_map').exists()


def test_subdirs_do_not_depend_on_existing_contents(tmp_path, monkeypatch):
    monkeypatch.setenv('OPGUARD_TRACE_OUTPUT_BASE', str(tmp_path / 'job'))
    for kind in ('exec_trace', 'callsite_trace'):
        expected = str(tmp_path / 'job' / kind)
        assert running.get_output_subdir(kind) == expected
        Path(expected).mkdir(parents=True, exist_ok=True)
        assert running.get_output_subdir(kind) == expected


def test_explicit_subdirectory_is_used_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv('OPGUARD_EXEC_OUTPUT_DIR', str(tmp_path / 'custom-exec'))
    assert running.get_output_subdir('exec_trace') == str(tmp_path / 'custom-exec')
    monkeypatch.setenv('OPGUARD_CALLSITE_OUTPUT_DIR', str(tmp_path / 'custom-calls'))
    assert running.get_output_subdir('callsite_trace') == str(tmp_path / 'custom-calls')
