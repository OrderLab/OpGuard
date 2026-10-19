"""An explicit launch contract. No command guessing or shell interpolation."""
import copy
import math
from pathlib import Path

RESERVED_ENV = {
    'OPGUARD_AUTOSTART', 'OPGUARD_CALLSITE_TRACER', 'OPGUARD_TRACE_OUTPUT_BASE',
    'OPGUARD_LOCAL_DUMP_TARGET', 'OPGUARD_LOCAL_DUMP_ENABLED', 'OPGUARD_LOCAL_DUMP_BASE',
    'OPGUARD_CALLSITE_OUTPUT_DIR', 'OPGUARD_EXEC_OUTPUT_DIR', 'OPGUARD_CONFIG_PATH',
    'OPGUARD_CALLSITE_CONFIG', 'OPGUARD_RUN_ID', 'OPGUARD_RUN_TOKEN', 'OPGUARD_TRACE_ROOT',
    'DEFAULT_CODE_REPO_BASE',
}


def strings(value, name, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value) or any(
            not isinstance(item, str) or '\0' in item for item in value):
        raise ValueError(f'{name} must be an argv list of strings')
    return value


def validate(spec):
    spec = copy.deepcopy(spec)
    allowed = {'baseline', 'candidate', 'backend', 'local_root', 'timeout_seconds',
               'max_stage_attempts', 'label', 'input_id'}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError('Unknown experiment fields')
    for key in ('backend', 'local_root', 'baseline', 'candidate'):
        if key not in spec:
            raise ValueError(f'Missing {key}')
    strings(spec['backend'], 'backend', True)
    local = Path(spec['local_root']).expanduser()
    if not local.is_absolute():
        raise ValueError('local_root must be an absolute node-local path')
    spec['local_root'] = str(local)
    for name in ('baseline', 'candidate'):
        workload = spec[name]
        if not isinstance(workload, dict) or set(workload) - {'cwd', 'launcher', 'script', 'args', 'env', 'ranks'}:
            raise ValueError(f'Unknown {name} workload fields')
        cwd = Path(workload['cwd']).expanduser().resolve()
        if not cwd.is_dir():
            raise ValueError(f'{name}.cwd does not exist')
        script = (cwd / workload['script']).resolve()
        if cwd not in script.parents or not script.is_file():
            raise ValueError(f'{name}.script must be a file inside its checkout')
        workload.update(cwd=str(cwd), script=str(script))
        strings(workload['launcher'], name + '.launcher', True)
        strings(workload.setdefault('args', []), name + '.args')
        env = workload.setdefault('env', {})
        if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                           or '=' in k or '\0' in k + v for k, v in env.items()):
            raise ValueError('env must contain string keys and values')
        if RESERVED_ENV & env.keys():
            raise ValueError('Output/lifecycle configuration belongs to the harness, not workload.env')
        ranks = workload.setdefault('ranks', [0])
        if not isinstance(ranks, list) or not ranks or any(type(r) is not int or r < 0 for r in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError('ranks must be distinct nonnegative integers')
        workload['ranks'] = sorted(ranks)
    if spec['baseline']['cwd'] == spec['candidate']['cwd']:
        raise ValueError('Use separate source checkouts for baseline and candidate')
    if spec['baseline']['ranks'] != spec['candidate']['ranks']:
        raise ValueError('v1 compares matching global ranks; rank remapping is not supported')
    timeout = spec.setdefault('timeout_seconds', 3600)
    attempts = spec.setdefault('max_stage_attempts', 2)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout_seconds must be positive and finite')
    if type(attempts) is not int or not 1 <= attempts <= 10:
        raise ValueError('max_stage_attempts must be between 1 and 10')
    return spec
