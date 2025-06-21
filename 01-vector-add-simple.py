"""
简化版向量加法测试 - 不包含性能基准测试
"""

import torch
import triton
import triton.language as tl

# 修复 Triton 3.1.0 API 兼容性
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.device.type == DEVICE.type and y.device.type == DEVICE.type and output.device.type == DEVICE.type
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# 测试代码
if __name__ == "__main__":
    print("开始测试 Triton 向量加法...")
    print(f"使用设备: {DEVICE}")
    
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    
    print(f"向量大小: {size}")
    
    # PyTorch 版本
    output_torch = x + y
    
    # Triton 版本
    output_triton = add(x, y)
    
    print("PyTorch 结果 (前5个元素):", output_torch[:5])
    print("Triton 结果 (前5个元素):", output_triton[:5])
    
    # 计算差异
    max_diff = torch.max(torch.abs(output_torch - output_triton))
    print(f'PyTorch 和 Triton 之间的最大差异: {max_diff}')
    
    if max_diff < 1e-6:
        print("✅ 测试通过！Triton 实现与 PyTorch 结果一致")
    else:
        print("❌ 测试失败！结果不一致")
