import sys
from types import SimpleNamespace

import pytest

from opguard import distributed as identity


@pytest.fixture(autouse=True)
def clean_identity(monkeypatch):
    for name in identity.GLOBAL_RANK_ENV + identity.LOCAL_RANK_ENV + identity.WORLD_SIZE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, 'torch.distributed', raising=False)


def test_single_process():
    assert identity.get_global_rank() == identity.get_local_rank() == 0
    assert identity.get_world_size() == 1


@pytest.mark.parametrize('name', identity.GLOBAL_RANK_ENV)
def test_scheduler_global_rank_wins_over_local(name, monkeypatch):
    monkeypatch.setenv(name, '9')
    monkeypatch.setenv('LOCAL_RANK', '1')
    assert identity.get_global_rank() == 9
    assert identity.get_local_rank() == 1


@pytest.mark.parametrize('name', identity.LOCAL_RANK_ENV)
def test_local_rank_aliases(name, monkeypatch):
    monkeypatch.setenv('RANK', '8')
    monkeypatch.setenv(name, '0')
    assert identity.get_local_rank() == 0
    assert identity.get_global_rank() == 8


def test_initialized_dist_has_priority(monkeypatch):
    monkeypatch.setenv('RANK', '2')
    monkeypatch.setitem(sys.modules, 'torch.distributed', SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: True,
        get_rank=lambda: 7, get_world_size=lambda: 16,
    ))
    assert identity.get_global_rank() == 7
    assert identity.get_world_size() == 16
    assert identity.get_local_rank() is None


def test_uninitialized_dist_uses_environment(monkeypatch):
    monkeypatch.setenv('RANK', '9')
    monkeypatch.setitem(sys.modules, 'torch.distributed', SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: False,
        get_rank=lambda: pytest.fail('must not query uninitialized process group'),
    ))
    assert identity.get_global_rank() == 9


@pytest.mark.parametrize('value', ['bad', '-1', ''])
def test_invalid_rank_uses_next_global_source(monkeypatch, value):
    monkeypatch.setenv('RANK', value)
    monkeypatch.setenv('SLURM_PROCID', '8')
    monkeypatch.setenv('LOCAL_RANK', '0')
    assert identity.get_global_rank() == 8


@pytest.mark.parametrize('env', [{'LOCAL_RANK': '0'}, {'LOCAL_RANK': '1'}, {'WORLD_SIZE': '16'}, {'RANK': 'bad'}])
def test_unknown_global_rank_is_not_zero(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match='global rank'):
        identity.get_global_rank()


def test_unknown_local_rank_is_not_global_rank(monkeypatch):
    monkeypatch.setenv('RANK', '8')
    monkeypatch.setenv('WORLD_SIZE', '16')
    assert identity.get_local_rank() is None


def test_runtime_metadata_uses_global_rank_before_init(monkeypatch):
    from opguard import runtime_tracer
    monkeypatch.setenv('SLURM_PROCID', '9')
    monkeypatch.setenv('LOCAL_RANK', '1')
    monkeypatch.setattr(runtime_tracer, '_mv', lambda: SimpleNamespace(get_meta=lambda rank: {'rank_seen': rank}))
    assert runtime_tracer._get_meta_info(10) == {'lineno': 10, 'rank_seen': 9}


def test_sync_aliases_use_same_rank(monkeypatch):
    from opguard import local_dump_manager
    monkeypatch.setenv('SLURM_PROCID', '9')
    monkeypatch.setenv('LOCAL_RANK', '1')
    assert local_dump_manager._get_gpu_rank() == 9
    assert local_dump_manager._get_process_rank() == 9


@pytest.mark.parametrize('mode', ['summary', 'full', 'both'])
def test_runtime_trace_and_full_tensor_use_same_global_rank(monkeypatch, mode):
    from opguard import runtime_tracer
    monkeypatch.setenv('RANK', '9')
    monkeypatch.setenv('LOCAL_RANK', '1')
    monkeypatch.setattr(runtime_tracer, '_maybe_torch', lambda: object())
    monkeypatch.setattr(runtime_tracer, 'serialize_tensor_io', lambda value, **kw: 'serialized')
    monkeypatch.setattr(runtime_tracer, '_mv', lambda: SimpleNamespace(get_meta=lambda rank: {'rank_seen': rank}))
    captured = []
    monkeypatch.setattr(runtime_tracer, '_emit_or_queue_trace', lambda *args: captured.append(args))
    runtime_tracer.save_tensor_trace('test', None, (), {}, None, dump_mode=mode, stack_trace_override='test')
    _, trace, full = captured[0]
    assert trace['gpu_rank'] == 9
    if mode == 'summary':
        assert full is None
    else:
        assert full['gpu_rank'] == full['meta']['rank_seen'] == 9


def test_metadata_injection_uses_shared_identity(monkeypatch):
    import libcst as cst
    from opguard.instrumentor.meta_var_injector import MetaVarInjector
    source = 'def forward(x):\n    return x\n'
    tree = cst.MetadataWrapper(cst.parse_module(source)).visit(MetaVarInjector({'function': 'forward'}))
    assert '_opguard_global_rank()' in tree.code
    assert 'dist.get_rank' not in tree.code
    compile(tree.code, '<instrumented>', 'exec')


def test_rank_import_does_not_load_torch():
    import subprocess
    result = subprocess.run([sys.executable, '-c',
        "import sys; import opguard.distributed; assert 'torch' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_second_node_local_leader_owns_shared_metadata(tmp_path):
    from opguard.local_sync.paths import ShadowPaths
    paths = ShadowPaths(tmp_path / 'shared', tmp_path / 'local')
    source = paths.files / 'instrumentation'
    source.mkdir()
    file = source / 'config.yaml'
    file.write_text('entry: []')
    assert file in list(paths.owned_files(rank=8, local_rank=0))
    assert file not in list(paths.owned_files(rank=9, local_rank=1))


@pytest.mark.parametrize('module', ['opguard.callsite_tracer._autostart', 'opguard.callsite_tracer.cupti_tracer._autostart'])
def test_autostart_uses_global_identity(module, monkeypatch):
    import importlib
    monkeypatch.setenv('SLURM_PROCID', '8')
    monkeypatch.setenv('LOCAL_RANK', '0')
    assert importlib.import_module(module)._rank_or_zero() == 8
