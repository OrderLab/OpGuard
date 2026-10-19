import functools
import json
import operator
import os
import subprocess
import sys

import pytest
import torch

from opguard.tracer import xor_signature as xs
from opguard.tracer.tensor_dump import summarize_tensor, finalize_summary
from opguard.trace_export.perfetto.scoring import compare_simple
from opguard.trace_export.perfetto.utils import io_summary_with_xor_values


def reference(t, regions=4):
    data = bytes(t.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8).cpu().tolist())
    words = [int.from_bytes(data[i:i+4], 'little') for i in range(0, len(data), 4)]
    chunks = [[] for _ in range(regions)]
    base, extra = divmod(len(words), regions)
    offset = 0
    for i in range(regions):
        length = base + (i < extra)
        chunks[i] = words[offset:offset+length]
        offset += length
    return [functools.reduce(operator.xor, chunk, 0) for chunk in chunks]


@pytest.mark.parametrize('size', [0, 1, 3, 4, 5, 7, 15, 16, 17, 108, 8193, 65539])
def test_all_bytes_and_empty_regions_cpu(size):
    t = torch.arange(size + 1, dtype=torch.int64).to(torch.uint8)[1:]
    for count in (1, 4, 16):
        assert xs.xor_signature(t, regional=count, by='words').tolist() == reference(t, count)
    assert xs.flat_word_layout(size)['byte_ranges'][-1][-1] == size


@pytest.mark.parametrize('t', [torch.tensor(3.), torch.arange(48.).reshape(6, 8).T,
                             torch.tensor([complex(1, 2), complex(-0., -3)]).conj(),
                             torch.tensor([0., -0., float('nan'), float('inf')]),
                             torch.arange(27, dtype=torch.bfloat16)])
def test_logical_bytes_cpu(t):
    assert xs.xor_signature(t, regional=4, by='words').tolist() == reference(t)


def test_diagnostic_default_and_cross_region_cancellation():
    a = torch.zeros(16, dtype=torch.int32)
    b = a.clone()
    b[1] = b[9] = 1
    assert xs.xor_signature(a).item() == xs.xor_signature(b).item()
    left, right = finalize_summary(summarize_tensor(a)), finalize_summary(summarize_tensor(b))
    assert left['stats']['xor_signature'] == [0, 0, 0, 0]
    assert right['stats']['xor_signature'] == [1, 0, 1, 0]
    assert xs.xor_signature(b.reshape(1, 4, 4), regional=4, by='words').tolist() == [1, 0, 1, 0]
    assert right['stats']['xor_sig_layout']['byte_ranges'] == [[0, 16], [16, 32], [32, 48], [48, 64]]
    assert 'outputs' in compare_simple({'outputs': left}, {'outputs': right}, False)[0]
    exported = io_summary_with_xor_values({'outputs': right})['outputs'][0]
    assert exported['xor_sig_layout'] == right['stats']['xor_sig_layout']
    assert json.loads(json.dumps(right)) == right


def test_same_region_collisions_remain_possible():
    t = torch.zeros(16, dtype=torch.int32)
    t[0] = t[1] = 1
    assert xs.xor_signature(t, regional=4, by='words').tolist() == [0, 0, 0, 0]


def test_meta_is_never_content_evidence():
    result = summarize_tensor(torch.empty(27, device='meta'))
    assert result['stats']['xor_sig_mode'] == 'placeholder-meta'
    assert 'xor_sig_layout' not in result['stats']


@pytest.mark.parametrize('count', ['1', '4'])
def test_environment_config_is_integer_and_applies_to_capture(count):
    code = "from opguard.tracer.tensor_dump import summarize_tensor; import torch; print(summarize_tensor(torch.ones(27))['stats']['xor_signature'].numel())"
    env = dict(os.environ, OPGUARD_XOR_REGIONAL=count)
    result = subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == count


def test_entry_handler_preserves_allocators_and_fill_policy(tmp_path):
    code = '''import torch
from opguard.tracer.entry_handler import enable_torch_deterministic_mode
torch.set_default_dtype(torch.float64)
torch.utils.deterministic.fill_uninitialized_memory = False
before = (torch.empty, torch.empty_like, torch.Tensor.new_empty)
enable_torch_deterministic_mode()
assert before == (torch.empty, torch.empty_like, torch.Tensor.new_empty)
assert torch.empty(0).dtype == torch.float64
assert torch.utils.deterministic.fill_uninitialized_memory is False
'''
    env = dict(os.environ, OPGUARD_TRACE_OUTPUT_BASE=str(tmp_path / 'capture'))
    subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True, text=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('size', [0, 1, 3, 4, 15, 108, 16384, 65539, 1048579])
@pytest.mark.parametrize('offset', [0, 1])
def test_native_gpu_all_bytes_and_unaligned_input(size, offset):
    assert hasattr(xs._xor_reduce_cuda_ext, 'xor_reduce_regional_cuda')
    t = torch.randint(0, 256, (size + offset,), dtype=torch.uint8, device='cuda')[offset:]
    for count in (1, 4, 16):
        assert xs.xor_signature(t, regional=count, by='words').cpu().tolist() == reference(t, count)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_gpu_fallback_and_logical_views(monkeypatch):
    monkeypatch.setattr(xs, '_xor_reduce_cuda_ext', None)
    for t in (torch.arange(27., device='cuda'), torch.arange(70., device='cuda').reshape(7, 10).T,
              torch.tensor(2., device='cuda'), torch.tensor([5, 7, 11], dtype=torch.uint8, device='cuda')):
        assert xs.xor_signature(t, regional=4, by='words').cpu().tolist() == reference(t)
        assert xs.xor_signature(t).cpu().tolist() == reference(t, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_native_on_two_current_streams_and_graph_capture():
    tensors, signatures = [], []
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        with torch.cuda.stream(stream):
            t = torch.arange(200000, dtype=torch.int32, device='cuda')
            tensors.append(t)
            signatures.append(xs.xor_signature(t, regional=4, by='words'))
    for stream in streams:
        stream.synchronize()
    assert all(s.cpu().tolist() == reference(t) for s, t in zip(signatures, tensors))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = xs.xor_signature(tensors[0], regional=4, by='words')
    graph.replay()
    assert out.cpu().tolist() == reference(tensors[0])
