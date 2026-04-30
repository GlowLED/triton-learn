import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(DEVICE)      # pyright: ignore[reportAttributeAccessIssue]
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

print(f"NUM_SM = {NUM_SM}\nNUM_REGS = {NUM_REGS}\nSIZE_SMEM = {SIZE_SMEM}\nWARP_SIZE = {WARP_SIZE}")


@triton.jit
def fused_softmax_kernel(
    input_ptr,
    output_ptr,
    input_stride_row,
    output_stride_row,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    start = tl.program_id(0)
    stride = tl.num_programs(0)
    
    for idx in tl.range(start, n_rows, stride, num_stages=num_stages):      # pyright: ignore[reportGeneralTypeIssues]
        row_input_start_ptr = input_ptr + idx * input_stride_row
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        row = tl.load(row_input_start_ptr + offsets, mask=mask, other=-float('inf'))
        
        row_minus_max = row - tl.max(row, axis=0)
        
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        
        row_softmax = numerator / denominator
        
        row_output_start_ptr = output_ptr + idx * output_stride_row
        
        tl.store(row_output_start_ptr + offsets, row_softmax, mask=mask)


kernels = {}
def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda
    
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    num_warps = 8
    
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    
    if kernel is None:
        kernel = fused_softmax_kernel.warmup(x, y, x.stride(0), y.stride(0), n_rows, n_cols,
                                    BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages, num_warps=num_warps, grid=(1,))
        kernel._init_handles()
        
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        
        occupany = min(NUM_REGS // (n_regs * WARP_SIZE * num_warps), SIZE_SMEM // size_smem)
        
        num_programs = occupany * NUM_SM
        
        kernels[BLOCK_SIZE] = (kernel, num_programs)
    
    num_programs = min(num_programs, n_rows)
        
    kernel[(num_programs, 1, 1)](
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_stages,
    )
    
    return y


torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda')
y_triton = fused_softmax(x)
y_torch = torch.softmax(x, dim=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
        
        
        

    
    
    
