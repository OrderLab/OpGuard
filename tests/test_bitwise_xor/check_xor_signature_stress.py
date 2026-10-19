import torch
from OpGuard.opguard.tracer.xor_signature import xor_signature
import random
import sys

def stress_test(shape, dtype, trials=10000):
    print(f"Stress-testing shape={shape}, dtype={dtype}, {trials} trials...")
    failures = 0

    for t in range(trials):
        # 1) build random x
        if dtype == torch.bfloat16:
            base = torch.randn(shape, device="cuda", dtype=torch.float32)
            x = base.to(torch.bfloat16)
        elif dtype.is_floating_point:
            x = torch.randn(shape, dtype=dtype, device="cuda")
        else:
            high = 256 if dtype==torch.uint8 else 10000
            x = torch.randint(0, high, shape, dtype=dtype, device="cuda")

        # 2) reproducibility
        s1 = xor_signature(x)
        s2 = xor_signature(x.clone())
        if s1 != s2:
            print(f"  ❌ reproducibility fail on trial {t}: {s1} vs {s2}")
            failures += 1
            if failures > 10: break
            continue

        # 3) sensitivity
        flat = x.view(-1)
        idx  = random.randrange(flat.numel())
        x2   = x.clone()
        if dtype.is_floating_point:
            # flip sign bit to guarantee a change
            flat2 = x2.view(-1)
            flat2[idx] = -flat2[idx]
        else:
            flat2 = x2.view(-1)
            flat2[idx] = flat2[idx] + 1

        s3 = xor_signature(x2)
        if s3 == s1:
            print(f"  ❌ sensitivity fail on trial {t}, idx {idx}: sig={s1}")
            failures += 1
            if failures > 10: break

    if failures == 0:
        print(f"✅ All {trials} trials passed for {dtype}, shape={shape}.")
    else:
        print(f"‼️  {failures} failures detected (see above).")
        sys.exit(1)

if __name__ == "__main__":
    # pick whatever combos you care about
    tests = [
        ((64,128,1024), torch.uint8),
        ((64,128,1024), torch.int32),
        ((64,128,1024), torch.bfloat16),
        ((64,128,1024), torch.float32),
    ]
    for shape, dt in tests:
        stress_test(shape, dt, trials=10000)
