"""Run outside the checkout after installing a native wheel, on a target GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def capture(backend):
    import torch
    if backend == "cupti":
        from opguard.callsite_tracer import opguard_cupti as tracer
        tracer.start_mem()
    else:
        from opguard.callsite_tracer.trace import _NativeCudaApi
        tracer = _NativeCudaApi()
        assert tracer.available
        tracer.start()
    try:
        x = torch.arange(256, dtype=torch.float32, device="cuda")
        y = (x.sin() * x).sum()
        torch.cuda.synchronize()
        tracer.flush()
        events = tracer.drain(10000)
        assert events, f"{backend} loaded but recorded no events"
        print(json.dumps({"backend": backend, "events": len(events), "result": y.item()}))
    finally:
        tracer.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", choices=("cupti", "cudaapi"))
    args = parser.parse_args()
    if args.capture:
        capture(args.capture)
        return
    import torch
    from opguard.doctor import inspect
    from opguard.tracer import xor_signature as xs
    report = inspect()
    assert Path(report["package"]).is_relative_to(Path(sys.prefix).resolve()), report["package"]
    assert report["native_ready"], report
    assert torch.cuda.is_available(), "A visible CUDA GPU is required for native validation"
    # Use native symbols directly: a fallback must not make this test pass.
    import xor_reduce_cuda as native
    assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), native.__file__
    checks = 0
    for n in (0, 1, 3, 17, 4097):
        data = (torch.arange(n, dtype=torch.int64) % 251).to(torch.uint8)
        device_data = data.cuda()
        actual = native.xor_reduce_cuda(device_data).view(torch.uint32).cpu()
        assert torch.equal(actual, xs.xor_signature(data)), n
        for regions in (1, 4, 8):
            actual = native.xor_reduce_regional_cuda(device_data, regions).view(torch.uint32).cpu()
            assert torch.equal(actual, xs.xor_signature(data, regional=regions, by="words")), (n, regions)
            checks += 1
    for backend in ("cupti", "cudaapi"):
        env = os.environ.copy()
        env["OPGUARD_AUTOSTART"] = "0"
        if backend == "cudaapi":
            hook = str(Path(report["package"]) / "callsite_tracer/native/libpylaunch_hook.so")
            env["LD_PRELOAD"] = " ".join(filter(None, (hook, env.get("LD_PRELOAD"))))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--capture", backend],
                       env=env, check=True, timeout=120)
    print(json.dumps({"native_xor_checks": checks + 5, "gpu": torch.cuda.get_device_name(), "verified": True}))


if __name__ == "__main__":
    main()
