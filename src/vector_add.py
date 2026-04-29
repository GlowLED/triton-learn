import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")
BLOCK_SIZE = 1024

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE) + start
    mask = offsets < N
    
    x_pos = x_ptr + offsets
    y_pos = y_ptr + offsets
    z_pos = z_ptr + offsets
    
    x = tl.load(x_pos, mask=mask)
    y = tl.load(y_pos, mask=mask)
    
    z = x + y
    
    tl.store(z_pos, z, mask=mask)

def add(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    z = torch.empty_like(x)
    
    assert x.is_cuda and y.is_cuda and z.is_cuda
    
    N = z.numel()
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    add_kernel[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)     # pyright: ignore[reportArgumentType]
    
    return z
 
def main():
    torch.manual_seed(42)
    x = torch.rand((1024*100 + 512,), dtype=torch.float32, device=DEVICE)
    y = torch.rand((1024*100 + 512,), dtype=torch.float32, device=DEVICE)
    z = add(x, y)
    print(z.allclose(x + y))
    
if __name__ == '__main__':
    main()
