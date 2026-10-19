import os
import math
import random
import torch

# 你自己的实现
from opguard.tracer.xor_signature import xor_signature

def cpu_ref_sig(x: torch.Tensor) -> int:
    """CPU 参考：小端打包成 u32 后 XOR 归约。与 CUDA 语义一致。"""
    b = x.contiguous().view(torch.uint8).cpu().numpy().ravel()
    acc = 0
    n = len(b)
    for i in range(0, n, 4):
        w = (int(b[i]) |
             (int(b[i+1]) << 8 if i+1 < n else 0) |
             (int(b[i+2]) << 16 if i+2 < n else 0) |
             (int(b[i+3]) << 24 if i+3 < n else 0))
        acc ^= w
    return acc & 0xFFFFFFFF

def assert_cuda_eq_cpu(x: torch.Tensor):
    sig_cuda = xor_signature(x).item()
    sig_cpu  = cpu_ref_sig(x)
    assert sig_cuda == sig_cpu, (sig_cuda, sig_cpu, x.dtype, x.shape, x.stride(), x.storage_offset())

def test_zero_int32_singleton():
    x = torch.zeros(1, dtype=torch.int32, device='cuda')
    assert xor_signature(x).item() == 0

def test_zero_various_sizes_and_dtypes():
    dtypes = [torch.int32, torch.float16, torch.bfloat16, torch.float32, torch.uint8, torch.int8]
    sizes  = [0, 1, 2, 3, 4, 5, 33, 127, 1024, 65537]  # 覆盖 Nbytes%4 各种尾数
    for dt in dtypes:
        for n in sizes:
            x = torch.zeros(n, dtype=dt, device='cuda')
            assert_cuda_eq_cpu(x)

def test_random_tensors_various_dtypes():
    torch.manual_seed(0)
    dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.uint8]
    for dt in dtypes:
        for n in [1, 3, 4, 5, 31, 32, 33, 12345]:
            x = torch.randn(n, dtype=dt, device='cuda') if dt.is_floating_point else torch.randint(0, 256, (n,), dtype=dt, device='cuda')
            assert_cuda_eq_cpu(x)

def test_bool_tensor_parity_property():
    # 纯 bool 时，只有 bit0/8/16/24 可能为 1
    torch.manual_seed(0)
    for n in [1, 2, 3, 4, 5, 33, 1234]:
        b = (torch.rand(n, device='cuda') > 0.5).to(torch.bool)
        sig = xor_signature(b).item()
        # 清除四个位后应为 0
        mask = (1 << 0) | (1 << 8) | (1 << 16) | (1 << 24)
        assert (sig & ~mask) == 0, hex(sig)
        assert_cuda_eq_cpu(b)

def test_misaligned_bf16_slices():
    # 常见：storage_offset % 4 == 2
    torch.manual_seed(123)
    x = torch.randn(1025, dtype=torch.bfloat16, device='cuda')
    # 1) 原始
    assert_cuda_eq_cpu(x)
    # 2) 去掉首元素，制造 offset=2(mod4)
    y = x[1:]
    assert y.storage_offset() % 2 == 1  # bfloat16 每元素2字节，byte offset=2(mod4)
    assert_cuda_eq_cpu(y)
    # 3) offset=1,3(mod4) 也过一遍
    z = x.view(torch.uint8)
    for off in [1, 3]:
        w = z[off:].view(torch.uint8)  # 任意字节对齐
        assert_cuda_eq_cpu(w)

def test_non_contiguous_and_strided_views():
    torch.manual_seed(42)
    m = torch.randn(64, 33, dtype=torch.float32, device='cuda')
    assert_cuda_eq_cpu(m)
    assert_cuda_eq_cpu(m.t())            # 转置
    assert_cuda_eq_cpu(m[:, ::2])        # 切列步幅
    assert_cuda_eq_cpu(m[::3, 5:])       # 行步幅+切片
    # 高维 flatten
    assert_cuda_eq_cpu(m.permute(1,0).contiguous().view(-1))

def test_large_inputs_crossing_many_blocks():
    # 选择能让 first-pass blocks 明显 > 256 的大小
    # BLOCK_SIZE=256, 每线程处理多个 word；保守取：words ~= 2**20
    nbytes = (1 << 22) + 123  # ~4MB+，尾数非4对齐
    x = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device='cuda')
    assert_cuda_eq_cpu(x)

def test_alignment_fastpath_vs_bytenpath_equivalence():
    # 构造 4 字节对齐且 Nbytes%4==0 的快路径
    n = 1_000_000
    x = torch.randint(0, 256, (n,), dtype=torch.uint8, device='cuda')
    # 通过 clone() 确保是基础对齐（PyTorch allocator 普遍 >=256B 对齐）
    a = x.clone()
    s1 = xor_signature(a).item()
    # 非对齐视图（+1字节），应自动走 byte-path，结果应一致于 CPU（语义无关对齐）
    b = a[1:]
    assert_cuda_eq_cpu(b)
    # 快路径和 CPU 的一致性
    assert s1 == cpu_ref_sig(a)

def test_empty_tensor_and_small_tails():
    for dt in [torch.float32, torch.int32, torch.uint8]:
        for n in [0, 1, 2, 3]:
            x = torch.zeros(n, dtype=dt, device='cuda')
            assert_cuda_eq_cpu(x)

def test_determinism_many_runs_same_tensor():
    torch.manual_seed(7)
    x = torch.randn(123456, dtype=torch.float16, device='cuda')[1:]  # 故意未对齐
    s = xor_signature(x).item()
    for _ in range(50):
        assert xor_signature(x).item() == s

def test_many_streams_concurrency():
    # 在多个 CUDA stream 下并发计算，验证使用当前流且无跨流读问题
    torch.manual_seed(0)
    streams = [torch.cuda.Stream() for _ in range(4)]
    xs = [torch.randn(200_003, dtype=torch.bfloat16, device='cuda')[1:] for _ in range(4)]
    sigs_cpu = [cpu_ref_sig(x) for x in xs]
    results = [None] * 4

    for i, st in enumerate(streams):
        with torch.cuda.stream(st):
            # 在各自流上启动 kernel
            results[i] = xor_signature(xs[i])

    # 切换回默认流，同步隐式发生在 .item()（PyTorch会同步依赖流）
    for i in range(4):
        assert results[i].item() == sigs_cpu[i]

def test_bool_pathological_parity():
    # 精心构造：在每个 mod4 分类中都让奇偶为 1，以触发 bit0/8/16/24 都为 1
    n = 68
    b = torch.zeros(n, dtype=torch.bool, device='cuda')
    b[0::4] = True
    b[1::4] = True
    b[2::4] = True
    b[3::4] = True
    sig = xor_signature(b).item()
    expected_mask = (1<<0)|(1<<8)|(1<<16)|(1<<24)
    assert sig == expected_mask, hex(sig)
    assert_cuda_eq_cpu(b)

def test_view_clone_consistency():
    torch.manual_seed(1234)
    x = torch.randn(34567, dtype=torch.float32, device='cuda')
    s0 = xor_signature(x).item()
    s1 = xor_signature(x.clone()).item()
    s2 = xor_signature(x[::1]).item()
    assert s0 == s1 == s2

def test_mix_host_device_roundtrip():
    # CPU→CUDA→签名 与 直接 CPU 参考一致
    torch.manual_seed(99)
    for n in [7, 31, 1009]:
        cpu = torch.randn(n, dtype=torch.float32)
        cu  = cpu.cuda()
        assert xor_signature(cu).item() == cpu_ref_sig(cpu)
