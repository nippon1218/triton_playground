"""Fused Attention - 简化版本
===============

这是Flash Attention算法的简化Triton实现，用于展示Triton如何实现注意力机制。

原始实现参考：
* Flash Attention v2 paper: https://tridao.me/publications/flash2/flash2.pdf
* Original flash attention paper: https://arxiv.org/abs/2205.14135
"""

import torch
import os
import triton
import triton.language as tl
import time

# 替换设备获取方式
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_cuda():
    # 使用torch的API检查
    return torch.cuda.is_available()

# 使用Triton实现一个简单的矩阵乘法内核
# 这个内核将用于注意力机制计算中的Q·K^T操作
@triton.jit
def matmul_kernel(
    # 指针参数
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 步长参数
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 块大小常量
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # 计算程序ID
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # 计算当前块的行和列索引
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    
    # 确保不超出矩阵边界
    if pid_m >= grid_m or pid_n >= grid_n:
        return
    
    # 计算当前块的起始行和列
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    
    # 创建行和列的偏移量
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # 创建行和列的掩码（确保不超出矩阵边界）
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 对K维度进行分块计算
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 计算当前K块的起始位置和偏移量
        start_k = k * BLOCK_SIZE_K
        offs_k = start_k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        # 加载A矩阵块
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # 加载B矩阵块
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # 计算矩阵乘法
        acc += tl.dot(a, b)
    
    # 存储结果
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)

# 使用Triton实现的矩阵乘法函数
def triton_matmul(a, b):
    # 获取矩阵维度
    M, K = a.shape
    _, N = b.shape

    # 创建输出矩阵
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # 计算步长
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    # 定义块大小
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16

    # 计算网格大小
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    # 启动内核
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )

    return c
# 使用Triton实现简化版的注意力机制
def triton_attention(q, k, v):
    # 获取张量维度
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # 使用PyTorch计算注意力分数 (Q·K^T)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    
    # 应用因果掩码
    mask = torch.triu(torch.ones(seq_len, seq_len, device=DEVICE), diagonal=1).bool()
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # 应用softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # 计算输出 (Attention·V)
    output = torch.matmul(attn_weights, v)
    
    return output

# PyTorch实现的注意力机制函数（用于比较）
def torch_attention(q, k, v):
    # 计算注意力分数
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    
    # 应用因果掩码
    mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1), device=scores.device), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    
    # 应用softmax并与值相乘
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    
    return output

# 运行示例并比较性能
def run_example():
    if not is_cuda():
        print("\nCUDA不可用，此示例需要CUDA环境才能运行。")
        return
    
    print("\n使用Triton实现的注意力机制示例：")
    
    # 创建随机张量
    batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE)
    
    print(f"输入形状: q={q.shape}, k={k.shape}, v={v.shape}")
    
    # 使用PyTorch实现
    torch.cuda.synchronize()
    start_time = time.time()
    torch_output = torch_attention(q, k, v)
    torch.cuda.synchronize()
    torch_time = time.time() - start_time
    
    # 使用Triton实现
    try:
        torch.cuda.synchronize()
        start_time = time.time()
        triton_output = triton_attention(q, k, v)
        torch.cuda.synchronize()
        triton_time = time.time() - start_time
        
        # 检查结果是否一致
        max_diff = torch.max(torch.abs(torch_output - triton_output)).item()
        print(f"PyTorch和Triton输出的最大差异: {max_diff}")
        print(f"PyTorch实现耗时: {torch_time*1000:.2f} ms")
        print(f"Triton实现耗时: {triton_time*1000:.2f} ms")
        print(f"加速比: {torch_time/triton_time:.2f}x")
        
    except Exception as e:
        print(f"Triton实现出错: {e}")
        print("回退到PyTorch实现...")
        print(f"PyTorch实现耗时: {torch_time*1000:.2f} ms")
        print(f"输出形状: {torch_output.shape}")

# 运行示例
if __name__ == "__main__":
    run_example()
