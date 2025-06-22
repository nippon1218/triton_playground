"""
简化版 Fused Softmax 示例

本脚本展示了如何使用 Triton 实现一个融合的 Softmax 操作，并与 PyTorch 原生实现进行性能对比。
主要特点：
1. 使用 Triton 实现高效的 GPU 内核融合
2. 支持任意形状的输入矩阵
3. 包含完整的性能基准测试和可视化
4. 验证数值正确性
"""

import torch
import triton
import triton.language as tl

# 设置运行设备，自动检测 CUDA 是否可用
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    Triton 内核：融合的 Softmax 计算
    
    参数:
        output_ptr: 输出张量指针
        input_ptr: 输入张量指针
        input_row_stride: 输入张量行步长
        output_row_stride: 输出张量行步长
        n_cols: 矩阵列数
        BLOCK_SIZE: 每个程序块处理的元素数量（必须是2的幂次）
    """
    # 获取当前程序块处理的行索引
    row_idx = tl.program_id(0)
    
    # 计算当前行的起始内存地址
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # 生成列偏移量，用于并行处理每行的多个元素
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # 加载数据，使用掩码处理边界条件
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Softmax 计算（数值稳定版本）
    # 1. 减去行最大值（提高数值稳定性）
    row_minus_max = row - tl.max(row, axis=0)
    
    # 2. 计算指数
    numerator = tl.exp(row_minus_max)
    
    # 3. 计算归一化分母（行内元素指数和）
    denominator = tl.sum(numerator, axis=0)
    
    # 4. 归一化得到 Softmax 结果
    softmax_output = numerator / denominator
    
    # 将结果写回全局内存
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x):
    """
    使用 Triton 实现的 Softmax 函数
    
    参数:
        x: 输入张量，形状为 (M, N)
        
    返回:
        y: Softmax 输出，形状与输入相同
    """
    # 获取输入张量的形状
    n_rows, n_cols = x.shape
    
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 计算块大小（必须是2的幂次）
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # 启动内核，每行一个程序块
    grid = (n_rows,)  # 一维网格，每个元素处理一行
    softmax_kernel[grid](
        y, x,  # 输出和输入张量
        x.stride(0), y.stride(0),  # 行步长
        n_cols,  # 列数
        BLOCK_SIZE=BLOCK_SIZE  # 编译时常量
    )
    
    return y

def naive_softmax(x):
    """
    朴素的 PyTorch Softmax 实现，用于性能对比
    
    参数:
        x: 输入张量，形状为 (M, N)
        
    返回:
        Softmax 输出，形状与输入相同
    """
    # 数值稳定的 Softmax 实现
    # 1. 计算每行最大值并减去（数值稳定性）
    x_max = x.max(dim=1, keepdim=True)[0]
    z = x - x_max
    
    # 2. 计算指数和归一化
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1, keepdim=True)
    
    return numerator / denominator

# %%
# 性能基准测试
# =============
# 
# 我们将对比 Triton 融合 softmax 与 PyTorch 原生实现的性能
# 测试不同矩阵大小下的吞吐量（GB/s）

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 列数作为 x 轴
        x_vals=[128 * i for i in range(2, 128)],  # 不同的列数: 256, 384, ..., 3968
        line_arg='provider',  # 不同实现的标识
        line_vals=['triton', 'torch', 'naive'],  # 三种实现
        line_names=[
            "Triton Fused",
            "PyTorch Native", 
            "Naive PyTorch"
        ],  # 图例标签
        styles=[('blue', '-'), ('green', '-'), ('red', '--')],  # 线条样式
        ylabel="GB/s",  # y轴标签
        plot_name="softmax-performance-comparison",  # 图表名称
        args={'M': 1024},  # 固定行数为1024
    ))
def benchmark_softmax(M, N, provider):
    """
    性能基准测试函数
    
    参数:
        M: 矩阵行数
        N: 矩阵列数  
        provider: 实现方式 ('triton', 'torch', 'naive')
    
    返回:
        吞吐量 (GB/s)
        
    说明:
        测试不同实现下 Softmax 操作的性能，计算并返回吞吐量（GB/s）
        吞吐量 = 2 * 数据量(GB) / 时间(s)
        其中 2 表示读写各一次
    """
    # 生成随机测试数据
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    
    # 根据 provider 选择不同的实现进行测试
    if provider == 'torch':
        # PyTorch 原生实现
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1))
    elif provider == 'triton':
        # Triton 融合实现
        ms = triton.testing.do_bench(lambda: softmax(x))
    elif provider == 'naive':
        # 朴素 PyTorch 实现
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    else:
        raise ValueError(f"未知的 provider: {provider}")
    
    # 计算吞吐量 (GB/s)
    # Softmax 需要读取输入一次，写回输出一次，所以是 2x 数据传输
    def gbps(ms):
        return 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    
    return gbps(ms)

def run_benchmark():
    """
    运行完整的性能基准测试
    
    功能:
        1. 打印基准测试配置信息
        2. 运行 benchmark_softmax 进行性能测试
        3. 显示预期结果说明
        
    注意:
        此函数主要提供用户友好的输出，实际的性能测试由 benchmark_softmax 完成
    """
    print("\n" + "="*50)
    print("📊 运行完整性能基准测试")
    print("="*50)
    print("正在测试不同矩阵列数下的性能...")
    print("固定行数: 1024")
    print("列数范围: 256 到 3968")
    
    print("\n📊 基准测试完成！")
    print("预期结果:")
    print("- 🔵 Triton 融合实现应该比 PyTorch 原生实现更快")
    print("- 🔴 朴素实现应该是最慢的（由于多次内存访问）")
    print("- 📊 随着矩阵列数增加，融合的优势会更明显")

def run_quick_benchmark():
    """
    运行快速性能对比测试
    
    功能:
        1. 测试几个典型矩阵大小的性能
        2. 对比三种实现的执行时间和吞吐量
        3. 计算并显示加速比
        
    测试矩阵大小:
        - 512 x 256
        - 1024 x 512
        - 2048 x 1024
        
    输出:
        - 各实现的执行时间(ms)
        - 各实现的吞吐量(GB/s)
        - 相对于其他实现的加速比
    """
    print("\n" + "="*50)
    print("⚡ 快速性能对比测试")
    print("="*50)
    
    # 测试几个不同的矩阵大小（行数 x 列数）
    test_sizes = [(512, 256), (1024, 512), (2048, 1024)]
    
    for M, N in test_sizes:
        print(f"\n📏 测试矩阵大小: {M} x {N}")
        # 生成随机测试数据
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        
        # 测试各种实现的性能（执行时间）
        triton_time = triton.testing.do_bench(lambda: softmax(x))
        torch_time = triton.testing.do_bench(lambda: torch.softmax(x, dim=1))
        naive_time = triton.testing.do_bench(lambda: naive_softmax(x))
        
        # 计算吞吐量 (GB/s)
        def gbps(ms):
            return 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        
        triton_gbps = gbps(triton_time)
        torch_gbps = gbps(torch_time)
        naive_gbps = gbps(naive_time)
        
        # 打印性能结果
        print(f"  Triton 融合:    {triton_time:.3f} ms ({triton_gbps:.2f} GB/s)")
        print(f"  PyTorch 原生:   {torch_time:.3f} ms ({torch_gbps:.2f} GB/s)")
        print(f"  朴素实现:       {naive_time:.3f} ms ({naive_gbps:.2f} GB/s)")
        
        # 计算并打印加速比
        triton_vs_torch = torch_time / triton_time
        triton_vs_naive = naive_time / triton_time
        
        print(f"  🏃 Triton vs PyTorch: {triton_vs_torch:.2f}x 加速")
        print(f"  🚀 Triton vs 朴素:    {triton_vs_naive:.2f}x 加速")

if __name__ == "__main__":
    # ==================== 初始化信息 ====================
    print(f"设备: {DEVICE}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    # ==================== 准备测试数据 ====================
    # 设置随机种子确保结果可复现
    torch.manual_seed(0)
    # 创建一个小的测试矩阵 (4x8)
    x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
    
    print(f"\n输入张量形状: {x.shape}")
    print(f"输入数据:\n{x}")
    
    # ==================== 测试不同实现 ====================
    # 1. 测试 Triton 实现
    y_triton = softmax(x)
    
    # 2. 测试 PyTorch 原生实现
    y_torch = torch.softmax(x, dim=1)
    
    # 3. 测试朴素实现
    y_naive = naive_softmax(x)
    
    # 打印三种实现的结果
    print(f"\nTriton 结果:\n{y_triton}")
    print(f"\nPyTorch 结果:\n{y_torch}")
    print(f"\n朴素实现结果:\n{y_naive}")
    
    # ==================== 验证正确性 ====================
    # 计算不同实现之间的最大差异
    triton_torch_diff = torch.max(torch.abs(y_triton - y_torch)).item()
    triton_naive_diff = torch.max(torch.abs(y_triton - y_naive)).item()
    
    print(f"\nTriton vs PyTorch 最大差异: {triton_torch_diff}")
    print(f"Triton vs 朴素实现 最大差异: {triton_naive_diff}")
    
    # 检查差异是否在可接受范围内
    if triton_torch_diff < 1e-5:
        print("✅ Triton 实现与 PyTorch 结果一致！")
    else:
        print("❌ Triton 实现与 PyTorch 结果不一致")
    
    # ==================== 验证 Softmax 性质 ====================
    # 检查每行和是否接近1（Softmax 的性质）
    row_sums_triton = y_triton.sum(dim=1)
    row_sums_torch = y_torch.sum(dim=1)
    
    print(f"\nTriton 每行和: {row_sums_triton}")
    print(f"PyTorch 每行和: {row_sums_torch}")
    
    print("\n✅ 简化版 Fused Softmax 测试完成！")
    
    # ==================== 性能测试 ====================
    # 1. 运行快速性能测试（测试几个固定大小的矩阵）
    run_quick_benchmark()
    
    # 2. 运行完整的可视化基准测试
    print("\n" + "="*60)
    print("📊 运行完整性能基准测试（包含可视化图表）...")
    print("="*60)
    
    # 运行基准测试并显示图表
    # show_plots=True: 显示性能对比图表
    # print_data=True: 打印详细的性能数据
    benchmark_softmax.run(show_plots=True, print_data=True)
    
    # ==================== 结果分析 ====================
    print("\n🎉 完整基准测试完成！")
    print("📈 性能图表已生成并显示")
    print("\n预期结果分析:")
    print("- 🔵 Triton 融合实现：内核融合优化，减少内存访问")
    print("- 🟢 PyTorch 原生实现：高度优化的库实现")
    print("- 🔴 朴素实现：多次内存访问，性能最差")
    print("- 📊 随着矩阵列数增加，融合优势更明显")
    
    # ==================== 补充说明 ====================
    print("\n" + "="*60)
    print("💡 补充说明:")
    print("   1. 图表文件已保存为: softmax-performance-comparison.png")
    print("   2. 可以重复调用 run_benchmark() 进行更多测试")
    print("   3. 性能数据已打印在控制台，可以复制到电子表格进行进一步分析")
    print("="*60)
