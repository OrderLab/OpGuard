"""AutoMetaVarInjector must not inject update_meta into Triton-decorated functions."""
import ast
import importlib.util
import sys
import uuid

import pytest

from opguard.instrumentor.auto_meta_var_injector import instrument_file_auto_meta_vars


def _instrument(tmp_path, source, name="mod.py"):
    path = tmp_path / name
    path.write_bytes(source.encode("utf-8"))
    original = path.read_bytes()
    return instrument_file_auto_meta_vars(str(path)), original, path.read_bytes()


def _injected_vars(source_bytes, func_name):
    tree = ast.parse(source_bytes.decode("utf-8"))
    (fn,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name]
    return sorted(kw.arg for c in ast.walk(fn) if isinstance(c, ast.Call)
                  and getattr(c.func, "id", None) == "update_meta" for kw in c.keywords)


KERNEL = """\
def kernel(x_ptr, n):
    batch_idx = tl.program_id(0)
    step = batch_idx * n
    tl.store(x_ptr + step, batch_idx)
"""
NESTED = """\
def kernel(x_ptr):
    def helper(v):
        step = v + 1
        return step
    class Inner:
        def method(self):
            global_step = 0
            global_step += 1
            return global_step
    batch_idx = tl.program_id(0)
    tl.store(x_ptr + batch_idx, helper(batch_idx))
"""


def case(case_id, imports, decorators, injected, body=KERNEL):
    source = f"import triton.language as tl\n{imports}\n\n\n" + "".join(f"{d}\n" for d in decorators) + body
    return pytest.param(source, injected, id=case_id)


CASES = [
    # Triton decorators, resolved through imports: never injected.
    case("triton.jit", "import triton", ["@triton.jit"], False),
    case("triton.jit(...)", "import triton", ["@triton.jit(do_not_specialize=['n'])"], False),
    case("from-triton-import-jit", "from triton import jit", ["@jit"], False),
    case("jit-as-tjit", "from triton import jit as tjit", ["@tjit"], False),
    case("triton-as-tr", "import triton as tr", ["@tr.jit"], False),
    case("triton-as-tr-call", "import triton as tr", ["@tr.jit()"], False),
    case("triton-as-t", "import triton as t", ["@t.jit"], False),
    case("autotune-on-jit", "import triton",
         ['@triton.autotune(configs=[triton.Config({"BLOCK": 64})], key=["n"])', "@triton.jit"], False),
    case("heuristics-on-jit", "import triton",
         ['@triton.heuristics({"EVEN": lambda args: args["n"] % 2 == 0})', "@triton.jit"], False),
    case("libentry-above-jit", "import triton\nfrom mylib import libentry", ["@libentry()", "@triton.jit"], False),
    case("libentry-below-jit", "import triton\nfrom mylib import libentry", ["@triton.jit", "@libentry"], False),
    case("triton.runtime-jit", "from triton.runtime import jit", ["@jit"], False),
    case("autotuner-autotune",
         "from triton import Config\nfrom triton.runtime import jit\nfrom triton.runtime.autotuner import autotune",
         ['@autotune(configs=[Config({"BLOCK": 32})], key=[])', "@jit"], False),
    case("autotuner-heuristics", "from triton.runtime import jit\nfrom triton.runtime.autotuner import heuristics",
         ['@heuristics({"B": lambda args: 16})', "@jit"], False),
    case("import-triton.runtime", "import triton.runtime", ["@triton.runtime.jit"], False),
    case("nested-def-and-class", "import triton", ["@triton.jit"], False, body=NESTED),
    case("star-jit", "from triton import *", ["@jit"], False),
    case("star-autotune-on-jit", "from triton import *", ["@autotune(configs=[], key=[])", "@jit"], False),
    # Same spelling, but not Triton after import resolution: still injected.
    case("local-def-jit", "def jit(f):\n    return f", ["@jit"], True),
    case("mylib-jit", "from mylib import jit", ["@jit"], True),
    case("segment-triton_utils", "from vllm import triton_utils", ["@triton_utils.jit"], True),
    case("segment-_triton", "from mypkg._triton import jit", ["@jit"], True),
    case("segment-triton_heuristics", "from mypkg.triton_heuristics import autotune", ["@autotune()"], True),
]


@pytest.mark.parametrize("source, injected", CASES)
def test_decorator_resolution(tmp_path, source, injected):
    result, original, after = _instrument(tmp_path, source)
    if injected:
        assert (result["status"], result.get("skipped_functions", 0)) == ("success", 0), result
        assert _injected_vars(after, "kernel") == ["batch_idx", "step"]
    else:
        assert (result["status"], result["skipped_functions"]) == ("no_changes", 1), result
        assert after == original


MIXED = "import functools\nimport torch\nimport triton\nimport triton.language as tl\n\n\n@triton.jit\n" + KERNEL + """

def train(n):
    global_step = 0
    for _ in range(n):
        step = 1
        global_step += 1
    return global_step

@functools.lru_cache(maxsize=None)
def cached(n):
    step = n
    return step

class Trainer:
    @torch.no_grad()
    def run(self):
        batch_idx = 3
        return batch_idx
"""


def test_regular_functions_in_same_file_still_injected(tmp_path):
    result, _, after = _instrument(tmp_path, MIXED)
    assert (result["status"], result["assignments_found"], result["skipped_functions"]) == ("success", 5, 1)
    assert _injected_vars(after, "kernel") == []
    assert _injected_vars(after, "train") == ["global_step", "global_step", "step"]
    assert _injected_vars(after, "cached") == ["step"]
    assert _injected_vars(after, "run") == ["batch_idx"]
    assert after.startswith(b"from opguard.tracer.meta_vars import update_meta\n")


def test_output_is_deterministic(tmp_path):
    assert _instrument(tmp_path, MIXED, "a.py")[2] == _instrument(tmp_path, MIXED, "b.py")[2]


def test_triton_only_file_is_byte_identical(tmp_path):
    source = ("# -*- coding: utf-8 -*-\r\nimport triton\r\nimport triton.language as tl\r\n\r\n"
              "@triton.jit\r\ndef kernel(x_ptr):  # trailing comment\r\n    batch_idx = tl.program_id(0)\r\n"
              "    micro_batch_idx = batch_idx; model_chunk_id = 0\r\n    tl.store(x_ptr + batch_idx, 1)\r\n")
    result, original, after = _instrument(tmp_path, source)
    assert (result["status"], result["assignments_found"]) == ("no_changes", 0), result
    assert after == original


def test_vllm_reexported_triton_not_injected(tmp_path):
    source = ("from vllm.triton_utils import tl, triton\nfrom vllm.triton_utils import triton as t\n\n\n"
              + "@triton.jit\n" + KERNEL.replace("kernel", "k1") + "\n\n"
              + "@triton.jit(do_not_specialize=['n'])\n" + KERNEL.replace("kernel", "k2") + "\n\n"
              + "@t.jit\n" + KERNEL.replace("kernel", "k3"))
    result, original, after = _instrument(tmp_path, source)
    assert (result["status"], result["skipped_functions"]) == ("no_changes", 3), result
    assert after == original


def test_import_after_kernel_definition(tmp_path):
    source = ("@tr.jit\n" + KERNEL + "\n\n@jit\n" + KERNEL.replace("kernel", "kernel_b")
              + "\n\ndef _late_imports():\n    from triton import jit\n\n\n"
              + "import triton as tr\nimport triton.language as tl\n")
    result, original, after = _instrument(tmp_path, source)
    assert (result["status"], result["skipped_functions"]) == ("no_changes", 2), result
    assert after == original


SMOKE = """\
import torch
import triton
import triton.language as tl

@triton.jit
def add_rows_kernel(x_ptr, y_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    batch_idx = tl.program_id(0)
    offs = batch_idx * n_cols + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < n_cols
    x, y = tl.load(x_ptr + offs, mask=mask), tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + 2.0 * y, mask=mask)

def launch(x, y):
    step = 0
    out = torch.empty_like(x)
    add_rows_kernel[(x.shape[0],)](x, y, out, x.shape[1], BLOCK=triton.next_power_of_2(x.shape[1]))
    step += 1
    return out
"""


def test_gpu_smoke_instrumented_triton_kernel_runs(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    result, _, _ = _instrument(tmp_path, SMOKE, "smoke_kernels.py")
    assert result["status"] == "success", result  # host-side `step` is still instrumented
    name = f"opguard_triton_smoke_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, str(tmp_path / "smoke_kernels.py"))
    module = sys.modules[name] = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        x, y = torch.randn(7, 45, device="cuda"), torch.randn(7, 45, device="cuda")
        out = module.launch(x, y)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x + 2.0 * y)
    finally:
        sys.modules.pop(name, None)
