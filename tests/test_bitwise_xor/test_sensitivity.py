import torch
import time
import matplotlib.pyplot as plt

import torch
from opguard.tracer.xor_signature import xor_signature

def benchmark(op_name, op_fn, x, warmup=5, repeat=10):
    # Warm-up
    for _ in range(warmup):
        _ = op_fn(x)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(repeat):
        start.record()
        _ = op_fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds

    avg_time = sum(times) / len(times)
    return avg_time

def speed():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sizes = [10**3, 10**4, 10**5, 10**6, 10**7, 10**8]
    print(f"{'Size':>10} | {'Sum (ms)':>10} | {'Std (ms)':>10} | {'Slowdown':>10}")
    print("-" * 46)

    for size in sizes:
        x = torch.randn(size, device=device)

        sum_time = benchmark("sum", torch.sum, x)
        std_time = benchmark("std", torch.std, x)

        slowdown = std_time / sum_time if sum_time > 0 else float('inf')
        print(f"{size:10,d} | {sum_time:10.4f} | {std_time:10.4f} | {slowdown:10.2f}×")

def sweep_mean_and_plot():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = 100_000
    delta = 0.01 
    torch.manual_seed(0)

    means = torch.logspace(start=-6, end=-1, steps=50)  # from 1e-6 to 0.1
    sum_changes = []
    std_changes = []
    bitwise_changes = []

    for mean in means:
        std_dev = 0.1 * mean.item()
        delta = mean.item()
        x = torch.normal(mean=mean.item(), std=std_dev, size=(size,), dtype=torch.float32, device=device)
        x = x.to(torch.bfloat16)
        x_mutated = x.clone()
        x_mutated[0:5] += delta

        # bfloat16 native sum and std
        sum_orig = x.sum().item()
        sum_mut = x_mutated.sum().item()
        std_orig = x.std().item()
        std_mut = x_mutated.std().item()
        bitwise_orig = xor_signature(x)
        bitwise_mut = xor_signature(x_mutated)

        sum_pct_change = 100 * abs(sum_mut - sum_orig) / sum_orig if sum_orig != 0 else 0
        std_pct_change = 100 * abs(std_mut - std_orig) / std_orig if std_orig != 0 else 0
        bitwise_pct_change = 0 if torch.equal(bitwise_orig, bitwise_mut) else 10

        sum_changes.append(sum_pct_change)
        std_changes.append(std_pct_change)
        bitwise_changes.append(bitwise_pct_change)

    plt.figure(figsize=(8, 5))
    plt.plot(means.cpu().numpy(), sum_changes, label="Sum % Change (bf16)", marker='o')
    plt.plot(means.cpu().numpy(), std_changes, label="Std % Change (bf16)", marker='x')
    plt.plot(means.cpu().numpy(), bitwise_changes, label="Bitwise % Change (bf16)", marker='^')

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mean Value (log scale)")
    plt.ylabel("Percentage Change (log scale)")
    plt.title("Sensitivity of Sum, Std, and Bitwise XOR Signature under Perturbation (bfloat16)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("bf16_sensitivity_plot.png", dpi=300)


if __name__ == "__main__":
    # main()
    sweep_mean_and_plot()
