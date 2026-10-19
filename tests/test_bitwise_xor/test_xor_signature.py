import torch
import random
import pytest

from typing import Callable

# Helper to compute XOR signature via CUDA kernel
# (redefines the logic to match your implementation)
from opguard.tracer.xor_signature import xor_signature

# Parametrized reproducibility & sensitivity tests
@pytest.mark.parametrize("shape", [
    (1,), (5,),
    (16, 16), (17, 13),
    (64, 128, 3),
])
@pytest.mark.parametrize("dtype, eps", [
    (torch.uint8,     1),
    (torch.int32,     1),
    (torch.bfloat16, 1),  # bit-level delta for BF16
    (torch.float32, 1e-6),
])
def test_reproducibility_and_sensitivity(shape, dtype, eps):
    torch.manual_seed(0)
    # Generate tensor
    if dtype == torch.bfloat16:
        x = torch.randn(shape, dtype=torch.float32, device="cuda").to(torch.bfloat16)
    elif dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype, device="cuda")
    else:
        # integer types
        high = 256 if dtype == torch.uint8 else 1000
        x = torch.randint(0, high, shape, dtype=dtype, device="cuda")

    # 1) reproducibility
    sig1 = xor_signature(x)
    sig2 = xor_signature(x.clone())
    assert sig1 == sig2, f"Signature mismatch on identical data: {sig1} vs {sig2}"

    # 2) sensitivity
    # pick a random element index
    flat = x.view(-1)
    idx = random.randrange(flat.numel())
    # make a copy
    x2 = x.clone()
    if dtype == torch.bfloat16:
        # bit-level bump: interpret bits as int16 and increment
        bits1 = flat.view(torch.int16)
        bits2 = x2.view(-1).view(torch.int16)
        assert bits2[idx] != bits1[idx] + 1, f"idx {idx} is {bits1[idx]}"
        print(f"Change {idx} from {bits1[idx]} to {bits1[idx] + 1}")
        bits2[idx] = bits1[idx] + 1
    else:
        flat2 = x2.view(-1)
        delta = eps if dtype.is_floating_point else 1
        assert flat2[idx] != flat2[idx] + delta, f"idx {idx} is {flat2[idx]}"
        print(f"Change {idx} from {flat2[idx]} to {flat2[idx] + delta}")
        flat2[idx] = flat2[idx] + delta

    

    sig3 = xor_signature(x2)
    assert sig3 != sig1, f"Signature did not change after perturbation at idx {idx}: {sig1} vs {sig3}"

# Edge-case tests

def test_all_zero_tensor():
    x = torch.zeros((1023,), dtype=torch.uint8, device="cuda")
    assert xor_signature(x) == 0


def test_all_same_value_tensor():
    for val in [0x00, 0xFF, 0x7F]:
        x = torch.full((1000,), val, dtype=torch.uint8, device="cuda")
        sig = xor_signature(x)
        # Even count -> 0, odd -> val
        expected = val if (1000 % 2 == 1) else 0
        assert sig == expected, f"Const-{val:#02x} tensor gave {sig:#02x}, expected {expected:#02x}"
