import torch

# pretend we have a [4092×6] bfloat16 tensor (6 bytes per row)
t = torch.randn(4092, 3, dtype=torch.bfloat16)

# this only reinterprets the dtype, it does *not* flatten:
b = t.contiguous().view(torch.uint8)
print(b.shape)   # ➔ torch.Size([4092, 6])

# now view as uint32 will fail, because 6 % 4 != 0
b.view(torch.uint32)
# RuntimeError: self.size(-1) must be divisible by 4 to view Byte as UInt32 … but got 6
