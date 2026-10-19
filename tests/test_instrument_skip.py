import os
import textwrap

from opguard.instrument_skip import (
    filter_instrumentation_entries,
    should_skip_instrument_path,
    should_skip_torch_nn_functional_call,
)
from opguard.instrumentor.op_wrapper.file_api import instrument_source_code_robust


def test_skip_torch_site_packages_path(monkeypatch):
    monkeypatch.setenv("OPGUARD_SKIP_PACKAGES", "custom_parallel,custom_optimizer")
    assert should_skip_instrument_path(
        "/usr/local/lib/python3.12/site-packages/torch/nn/functional.py"
    )
    assert should_skip_instrument_path(
        "/usr/local/lib/python3.12/site-packages/torchgen/gen.py"
    )
    assert not should_skip_instrument_path(
        "/workspace/model/megatron/model/transformer.py"
    )
    assert not should_skip_instrument_path(
        "/workspace/model/modeling.py"
    )
    assert should_skip_instrument_path(
        "/usr/local/lib/python3.12/site-packages/custom_parallel/parallel/fsdp2/_dbuffer_api.py"
    )
    assert should_skip_instrument_path(
        "/usr/local/lib/python3.12/site-packages/custom_parallel/dtensor/torch_original/pt27/_api.py"
    )
    assert not should_skip_instrument_path(
        "/workspace/custom_parallel/parallel/fsdp2/_dbuffer_api.py"
    )
    assert should_skip_instrument_path(
        "/usr/local/lib/python3.12/site-packages/custom_optimizer/base_optimizer/compiled_optimizer.py"
    )
    assert not should_skip_instrument_path(
        "/usr/local/lib/python3.12/site-packages/custom_checkpoint/checkpointer/base_checkpointer.py"
    )


def test_instrument_torch_override(monkeypatch):
    monkeypatch.setenv("OPGUARD_INSTRUMENT_TORCH", "1")
    assert not should_skip_instrument_path(
        "/usr/local/lib/python3.12/site-packages/torch/nn/functional.py"
    )


def test_skip_torch_nn_functional_dotted():
    assert should_skip_torch_nn_functional_call("torch.nn.functional.dropout")
    assert should_skip_torch_nn_functional_call("F.dropout")
    assert not should_skip_torch_nn_functional_call("self.dropout")
    assert not should_skip_torch_nn_functional_call("rmsnorm_backward")
    assert not should_skip_torch_nn_functional_call("torch.nn.functional.cross_entropy")


def test_filter_drops_configured_packages_keeps_user(monkeypatch):
    monkeypatch.setenv("OPGUARD_SKIP_PACKAGES", "custom_parallel,custom_optimizer")
    kept, skipped = filter_instrumentation_entries([
        {"file_path": "/usr/local/lib/python3.12/site-packages/torch/nn/functional.py", "function": "dropout_"},
        {"file_path": "/usr/local/lib/python3.12/site-packages/custom_parallel/kernels/zero.py", "function": "zero_"},
        {"file_path": "/usr/local/lib/python3.12/site-packages/custom_optimizer/base_optimizer/compiled_optimizer.py", "function": "step"},
        {"file_path": "/workspace/model/models/foo.py", "function": "forward"},
    ])
    assert len(kept) == 1
    assert kept[0]["function"] == "forward"
    assert skipped == [
        "/usr/local/lib/python3.12/site-packages/torch/nn/functional.py",
        "/usr/local/lib/python3.12/site-packages/custom_parallel/kernels/zero.py",
        "/usr/local/lib/python3.12/site-packages/custom_optimizer/base_optimizer/compiled_optimizer.py",
    ]


def test_does_not_wrap_dropout_inside_jit_script():
    src = textwrap.dedent("""
    import torch

    def bias_dropout_add(x, bias, residual, prob, training):
        out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
        return residual + out

    @torch.jit.script
    def bias_dropout_add_fused_train(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, True)
    """)
    config = [
        {"function": "dropout", "tag_names": ["x"]},
        {"function": "bias_dropout_add", "tag_names": ["x"]},
    ]
    out = instrument_source_code_robust(src, config)
    assert "opguard_wrapped" not in out
    assert "@torch.jit.script" in out
    assert "torch.nn.functional.dropout" in out


def test_custom_packages_are_opt_in(monkeypatch):
    monkeypatch.delenv("OPGUARD_SKIP_PACKAGES", raising=False)
    path = "/env/site-packages/custom_optimizer/step.py"
    assert not should_skip_instrument_path(path)
    monkeypatch.setenv("OPGUARD_SKIP_PACKAGES", " custom_optimizer ,other")
    assert should_skip_instrument_path(path)
    assert not should_skip_instrument_path("/workspace/custom_optimizer/step.py")
