# simple_triton_add.py
import torch
import triton
import triton.language as tl

# --- Triton kernel: adds A and B into C --------------------------------------
@triton.jit
def vec_add_kernel(A_ptr, B_ptr, C_ptr, N,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                         # which block am I
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # element indices
    mask = offs < N

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offs, c, mask=mask)

# --- PyTorch wrapper ----------------------------------------------------------
def triton_vec_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.shape == B.shape, "Shapes must match"
    assert A.dtype == B.dtype == torch.float32, "Use float32 for this demo"

    N = A.numel()
    C = torch.zeros_like(A)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    vec_add_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)
    return C

# --- Demo --------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    N = 1_000_123
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)

    C = triton_vec_add(A, B)
    # quick correctness check vs PyTorch
    torch.testing.assert_close(C, A + B)
    print("OK! Triton kernel produced the correct result.")
