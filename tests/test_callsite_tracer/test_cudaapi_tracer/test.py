# test.py
import torch

def do_mm(n=2048):
    a = torch.randn((n, n), device="cuda")
    b = torch.randn((n, n), device="cuda")
    return a @ b

def main():
    # Warm-up
    for _ in range(2):
        do_mm(512); torch.cuda.synchronize()
    # A few launches that will show distinct Python stacks
    c1 = do_mm(1024); torch.cuda.synchronize()
    c2 = do_mm(1536); torch.cuda.synchronize()

if __name__ == "__main__":
    main()
