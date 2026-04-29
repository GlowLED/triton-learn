import torch
import triton
import triton.language as tl

DEVICE = "cuda:0"

@triton.jit
def fused_softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    
    for idx in range(pid, n_rows, stride):
        input_row_start = input_ptr + idx * n_cols
        
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        input_row = tl.load(input_row_start + offsets, mask=mask, other=-float('inf'))
        minus_max_row = input_row - tl.max(input_row)
        
        numerator = tl.exp(minus_max_row)
        denominator = tl.sum(numerator)
        
        output_softmax = numerator / denominator
        output_row_start = output_ptr + idx * n_cols
        tl.store(output_row_start + offsets, output_softmax, mask=mask)
        
        
def fused_softmax(input: torch.Tensor):
    output = torch.empty_like(input)
    assert input.is_cuda and output.is_cuda
    
    n_rows, n_cols = input.size()
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    fused_softmax_kernel[(128,)](input, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return output
    

if __name__ == '__main__':
    x = torch.rand((128*10 + 64, 100), dtype=torch.float32, device=DEVICE)
    y = fused_softmax(x)
    print(y.allclose(torch.softmax(x, dim=1)))
    
    
        
        