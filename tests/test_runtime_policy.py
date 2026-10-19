"""Check real interpreter startup and numerical state, not only helper functions."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import opguard


def run_child(code, tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPGUARD_")}
    env.update(OPGUARD_AUTOSTART="0", OPGUARD_DETERMINISTIC="0",
               OPGUARD_CALLSITE_TRACER="0", OPGUARD_AUTO_INSTRUMENT="0",
               OPGUARD_TRACE_OUTPUT_BASE=str(tmp_path / "capture"),
               PYTHONPATH=str(Path(opguard.__file__).resolve().parent.parent))
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("entry", ["startup", "injected_entry"])
def test_tracing_preserves_allocators_rng_and_numerical_settings(tmp_path, entry):
    run_child('''
import os, runpy, torch
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.utils.deterministic.fill_uninitialized_memory = False
torch.set_default_dtype(torch.float64)
before = (torch.empty, torch.empty_like, torch.Tensor.new_empty)
rng = torch.get_rng_state().clone()
os.environ.update(OPGUARD_AUTOSTART="1", OPGUARD_PATCH_EMPTY="1", OPGUARD_FILL_UNINITIALIZED="1")
''' + ('''runpy.run_module("usercustomize")
''' if entry == "startup" else '''from opguard.tracer.entry_handler import _opguard_entry_handler
_opguard_entry_handler()
''') + '''
assert before == (torch.empty, torch.empty_like, torch.Tensor.new_empty)
assert torch.equal(rng, torch.get_rng_state())
assert not torch.are_deterministic_algorithms_enabled()
assert torch.backends.cudnn.benchmark and not torch.backends.cudnn.deterministic
assert torch.backends.cuda.matmul.allow_tf32
assert not torch.utils.deterministic.fill_uninitialized_memory
assert torch.empty(0).dtype == torch.float64
from pathlib import Path
assert not (Path(os.environ["OPGUARD_TRACE_OUTPUT_BASE"]) / "runtime_policy").exists()
''', tmp_path)


@pytest.mark.parametrize("deferred", [False, True])
def test_explicit_intervention_is_recorded_without_allocator_replacement(tmp_path, deferred):
    prefix = '''import os, runpy, json
from pathlib import Path
'''
    if not deferred:
        prefix += '''import torch
torch.utils.deterministic.fill_uninitialized_memory = False
before = (torch.empty, torch.empty_like, torch.Tensor.new_empty)
'''
    run_child(prefix + '''
os.environ["OPGUARD_DETERMINISTIC"] = "1"
runpy.run_module("usercustomize")
import torch
assert torch.are_deterministic_algorithms_enabled()
assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
assert not torch.backends.cuda.matmul.allow_tf32
records = list((Path(os.environ["OPGUARD_TRACE_OUTPUT_BASE"]) / "runtime_policy").glob("*.json"))
assert len(records) == 1
record = json.loads(records[0].read_text())
assert record["reason"] == "OPGUARD_DETERMINISTIC=1"
assert record["allocator_functions_replaced"] is False
assert record["before"]["fill_uninitialized_memory"] == record["after"]["fill_uninitialized_memory"]
assert record["after"]["deterministic_algorithms"] is True
assert torch.empty.__name__ == "empty"
from opguard.runtime_policy import apply_requested_policy
apply_requested_policy()
assert len(list(records[0].parent.glob("*.json"))) == 1
''' + ('' if deferred else '''
assert before == (torch.empty, torch.empty_like, torch.Tensor.new_empty)
assert not torch.utils.deterministic.fill_uninitialized_memory
'''), tmp_path)
