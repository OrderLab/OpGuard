import pytest
import torch

from opguard.callsite_tracer.trace import CallsiteTracer
from opguard.trace_export.perfetto.scoring import compare_simple


def test_cpu_discovery_is_opt_in_and_tracks_nested_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("OPGUARD_CALLSITE_INCLUDE_CPU", "1")
    with CallsiteTracer(dump_every=0, dump_path=str(tmp_path / "calls.json"),
                        enable_cupti=False, enable_cudaapi=False) as tracer:
        x = torch.arange(7)
        torch.cat([x, x])
    ops = {op for site in tracer.stats["python"].values() for op in site["ops"]}
    assert "aten.arange.default" in ops
    assert "aten.cat.default" in ops
    monkeypatch.delenv("OPGUARD_CALLSITE_INCLUDE_CPU")
    with CallsiteTracer(dump_every=0, dump_path=str(tmp_path / "off.json"),
                        enable_cupti=False, enable_cudaapi=False) as tracer:
        torch.arange(7)
    assert tracer._total_all == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_factories_and_tensor_lists_are_discovered(tmp_path):
    with CallsiteTracer(dump_every=0, dump_path=str(tmp_path / "calls.json"),
                        enable_cupti=False, enable_cudaapi=False) as tracer:
        x = torch.arange(7, device="cuda")
        torch.cat([x, x])
    ops = {op for site in tracer.stats["python"].values() for op in site["ops"]}
    assert "aten.arange.default" in ops
    assert "aten.cat.default" in ops


def test_checkout_location_is_not_a_tensor_difference():
    a = {"function": "f", "tag_prefix": "/checkout/a/model.py", "outputs": 5}
    b = {**a, "tag_prefix": "/checkout/b/model.py"}
    assert not compare_simple(a, b)[0]
    assert compare_simple(a, {**b, "outputs": 6})[0]
