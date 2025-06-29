# 简化版分组矩阵乘法示例
# 基于Triton 3.1.0版本

import torch
import triton
import triton.language as tl
import time

# 使用PyTorch的API获取设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 简单的Triton内核示例 - 元素级操作
@triton.jit
def add_kernel(
    x_ptr,  # 输入张量指针
    y_ptr,  # 输出张量指针
    n_elements,  # 元素数量
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    # 计算程序ID和偏移量
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建掩码以处理边界情况
    mask = offsets < n_elements
    
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 执行计算 (简单地将值加倍)
    y = x * 2.0
    
    # 存储结果
    tl.store(y_ptr + offsets, y, mask=mask)

# 使用简单的Triton内核示例
def run_triton_example():
    print("运行简单的Triton内核示例...")
    
    # 创建输入张量
    n_elements = 1024
    x = torch.rand(n_elements, device=DEVICE, dtype=torch.float32)
    y = torch.empty_like(x)
    
    # 计算网格大小
    block_size = 128
    grid = (triton.cdiv(n_elements, block_size),)
    
    # 使用PyTorch计算参考结果
    y_ref = x * 2.0
    
    # 使用Triton计算结果
    try:
        add_kernel[grid](x, y, n_elements, BLOCK_SIZE=block_size)
        
        # 验证结果
        max_diff = torch.max(torch.abs(y - y_ref)).item()
        print(f"PyTorch和Triton结果的最大差异: {max_diff}")
        
        # 性能测试
        n_repeat = 100
        torch_start = time.time()
        for _ in range(n_repeat):
            _ = x * 2.0
        torch.cuda.synchronize()
        torch_end = time.time()
        torch_time = (torch_end - torch_start) * 1000 / n_repeat  # 转换为毫秒
        
        triton_start = time.time()
        for _ in range(n_repeat):
            add_kernel[grid](x, y, n_elements, BLOCK_SIZE=block_size)
        torch.cuda.synchronize()
        triton_end = time.time()
        triton_time = (triton_end - triton_start) * 1000 / n_repeat  # 转换为毫秒
        
        print(f"PyTorch实现耗时: {torch_time:.4f} ms")
        print(f"Triton实现耗时: {triton_time:.4f} ms")
        print(f"加速比: {torch_time / triton_time:.2f}x")
        
    except Exception as e:
        print(f"Triton实现出错: {e}")

# 使用PyTorch实现的分组矩阵乘法
def run_pytorch_gemm_example():
    print("\n运行PyTorch分组矩阵乘法示例...")
    
    # 创建一组矩阵
    group_size = 4
    group_A = []
    group_B = []
    group_C = []
    
    # 创建不同大小的矩阵
    sizes = [(128, 64), (256, 128), (64, 32), (512, 256)]
    
    for i in range(group_size):
        M, K = sizes[i]
        A = torch.rand((M, K), device=DEVICE, dtype=torch.float32)
        B = torch.rand((K, M), device=DEVICE, dtype=torch.float32)  # 使N=M简化测试
        group_A.append(A)
        group_B.append(B)
    
    # 使用PyTorch计算结果
    start_time = time.time()
    for i in range(group_size):
        C = torch.matmul(group_A[i], group_B[i])
        group_C.append(C)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"PyTorch分组矩阵乘法耗时: {(end_time - start_time) * 1000:.2f} ms")

if __name__ == "__main__":
    run_triton_example()
    run_pytorch_gemm_example()
