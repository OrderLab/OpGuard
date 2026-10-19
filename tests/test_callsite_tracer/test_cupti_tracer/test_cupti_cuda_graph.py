# demo_cudagraph_cupti.py
import os, time
import torch

# Ensure your C++ bridge was built/installed as "opguard_cupti" (from your .cpp)
import opguard.callsite_tracer.opguard_cupti as cupti

# 1) Start CUPTI in memory mode (callbacks + activity enabled by your bridge)
#    Keep Python stack capture ON (your bridge defaults to on; make explicit here):
# os.environ["OPGUARD_STACKS"] = "1"
# os.environ["OPGUARD_DEBUG"] = "1"
cupti.start_mem()

# 2) Make a tiny workload with CUDA Graphs
torch.cuda.synchronize()
torch.manual_seed(0)

M, K, N = 1024, 1024, 1024
a_static = torch.empty(M, K, device='cuda', dtype=torch.float32).normal_()
b_static = torch.empty(K, N, device='cuda', dtype=torch.float32).normal_()
out_static = torch.empty(M, N, device='cuda', dtype=torch.float32)

stream = torch.cuda.Stream()
g = torch.cuda.CUDAGraph()

# --- Warmup on the SAME stream (pre-capture) ---
with torch.cuda.stream(stream):
    # same shapes/dtypes as in capture
    torch.mm(a_static, b_static, out=out_static)   # forces cuBLAS/Lt init & algo pick
    torch.relu_(out_static)                         # optional warmup
torch.cuda.synchronize()

# --- Capture on that stream ---
torch.cuda.set_stream(stream)
with torch.cuda.graph(g, stream=stream):
    torch.mm(a_static, b_static, out=out_static)   # now capturable
    torch.relu_(out_static)                        # capturable

# --- Replay (just to prove it runs) ---
g.replay()
torch.cuda.synchronize()

# 5) Flush CUPTI and drain events
cupti.flush()
events = cupti.drain(100000)

# 6) Show a tiny summary: kernel name + lowest user frame (file:line func)
def lowest_user_frame(stack):
    for file, line, func in reversed(stack):
        if ("/torch/" in file) or ("site-packages/torch" in file) or ("opguard_cupti" in file):
            continue
        return f"{file}:{line} in {func}"
    return "<no-python-frame>"

print(f"Drained {len(events)} kernel activity records")
seen = 0
for ev in events:
    kernel = ev.get("kernel")
    stack = ev.get("stack") or []
    src = lowest_user_frame(stack)
    print(f"{kernel:60s}  ←  {src}")
    seen += 1
    # if seen >= 12:  # keep the demo short
    #     break

cupti.stop()
