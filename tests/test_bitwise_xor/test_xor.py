import torch# This is the name from setup.py
import time
from OpGuard.opguard.tracer.xor_signature import xor_signature

def benchmark(name, fn, warmup=3, repeat=100):
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.time()
        _ = fn()
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

    avg_time_ms = 1000 * sum(times) / len(times)
    print(f"{name}: {avg_time_ms:.4f} ms")

if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn((4096, 2048), dtype=torch.bfloat16, device="cuda")

    def sum_fn():
        return x.sum()

    def mean_fn():
        return x.mean()

    def std_fn():
        return x.std()

    def min_fn():
        return x.min()

    def max_fn():
        return x.max()
    

    def xor_fn():
        return xor_signature(x)

    benchmark("sum()", sum_fn)
    benchmark("mean()", mean_fn)
    benchmark("std()", std_fn)
    benchmark("min()", min_fn)
    benchmark("max()", max_fn)
    benchmark("xor_signature()", xor_fn)

