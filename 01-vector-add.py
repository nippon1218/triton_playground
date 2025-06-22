"""
向量加法示例
===========

本教程展示如何使用Triton实现简单的向量加法。

通过本教程，您将学习到：

* Triton的基本编程模型
* 使用`triton.jit`装饰器定义Triton内核
* 如何验证自定义操作的正确性
* 如何对自定义操作进行基准测试，与原生实现进行性能对比
"""

# %%
# 1. 计算内核
# --------------
import torch
import triton
import triton.language as tl  # Triton语言扩展，提供GPU编程原语

# 设置计算设备：如果CUDA可用则使用GPU，否则使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@triton.jit
def add_kernel(x_ptr,  # 第一个输入向量的指针
               y_ptr,  # 第二个输入向量的指针
               output_ptr,  # 输出向量的指针
               n_elements,  # 向量的大小
               BLOCK_SIZE: tl.constexpr,  # 每个程序块处理的元素数量
               ):
    """
    Triton内核函数：执行向量加法
    
    参数:
        x_ptr, y_ptr: 输入向量的设备内存指针
        output_ptr: 输出向量的设备内存指针
        n_elements: 向量中的元素总数
        BLOCK_SIZE: 每个线程块处理的元素数量
    """
    # 获取当前程序(线程块)的ID
    pid = tl.program_id(axis=0)  # 使用1D网格，所以axis=0
    
    # 计算当前线程块处理的数据范围
    # 例如：向量长度为256，BLOCK_SIZE=64，则4个线程块分别处理：
    # 线程块0: [0:64], 线程块1: [64:128], 线程块2: [128:192], 线程块3: [192:256]
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建掩码，防止内存越界访问
    # 当输入大小不是BLOCK_SIZE的整数倍时，最后一个线程块可能不需要处理所有元素
    mask = offsets < n_elements
    
    # 从全局内存加载数据到寄存器
    x = tl.load(x_ptr + offsets, mask=mask)  # 加载x向量的数据
    y = tl.load(y_ptr + offsets, mask=mask)  # 加载y向量的数据
    
    # 执行向量加法
    output = x + y
    
    # 将结果写回全局内存
    tl.store(output_ptr + offsets, output, mask=mask)


# %%
# 2. 辅助函数：准备数据并调用Triton内核
# -----------------------------------------

def add(x: torch.Tensor, y: torch.Tensor):
    """
    执行向量加法的Python辅助函数
    
    参数:
        x, y: 输入张量
    返回:
        output: x + y的结果张量
    """
    # 预分配输出张量
    output = torch.empty_like(x)
    
    # 确保所有张量都在正确的设备上
    assert x.device.type == DEVICE.type and y.device.type == DEVICE.type and output.device.type == DEVICE.type
    
    # 获取元素总数
    n_elements = output.numel()
    
    # 定义计算网格大小（即需要启动多少个线程块）
    # triton.cdiv(a, b) 计算a除以b的向上取整
    # 这样确保即使元素数量不是BLOCK_SIZE的整数倍，也能处理所有元素
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # 调用Triton内核
    # 注意：
    # 1. 使用[grid]语法指定网格大小
    # 2. 张量会自动转换为指向其第一个元素的指针
    # 3. BLOCK_SIZE作为关键字参数传递
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    # 注意：此时内核可能仍在异步执行
    # 如果需要确保计算完成，可以调用torch.cuda.synchronize()
    return output


# %%
# 3. 验证正确性
# ----------------

# 设置随机种子以确保结果可重现
torch.manual_seed(0)

# 定义向量大小
size = 98432  # 可以修改为任意大小，不一定是BLOCK_SIZE的倍数

# 生成随机输入数据
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)

# 使用PyTorch原生实现计算参考结果
output_torch = x + y

# 使用Triton实现计算结果
output_triton = add(x, y)

# 打印结果（前几个元素）
print("PyTorch结果:", output_torch[:5])
print("Triton结果: ", output_triton[:5])

# 计算并打印最大差异（应该非常接近0）
max_diff = torch.max(torch.abs(output_torch - output_triton))
print(f'PyTorch和Triton实现之间的最大差异: {max_diff.item():.6f}')

# %%
# Seems like we're good to go!

# %%
# 4. 性能基准测试
# ----------------
# 现在我们对不同大小的向量进行基准测试，比较Triton实现与PyTorch原生实现的性能

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # 作为x轴的参数名
        x_vals=[2**i for i in range(12, 28, 1)],  # 测试的向量大小范围：2^12 到 2^27
        x_log=True,  # x轴使用对数刻度
        line_arg='provider',  # 用于区分不同曲线的参数名
        line_vals=['triton', 'torch'],  # 测试两种实现
        line_names=['Triton', 'Torch'],  # 图例名称
        styles=[('blue', '-'), ('green', '-')],  # 线条样式
        ylabel='GB/s',  # y轴标签：吞吐量(GB/s)
        plot_name='vector-add-performance',  # 图表名称，也用作保存文件的名称
        args={},  # 其他固定参数
    ))
def benchmark(size, provider):
    """
    性能基准测试函数
    
    参数:
        size: 向量大小
        provider: 实现方式，'triton'或'torch'
    返回:
        (中位数吞吐量, 最小吞吐量, 最大吞吐量)
    """
    # 准备测试数据
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    
    # 定义分位数用于计算性能指标
    quantiles = [0.5, 0.2, 0.8]  # 中位数、20%和80%分位数
    
    # 测试PyTorch原生实现
    if provider == 'torch':
        # 中位时间：对应分位数0.5
        # 20%时间：对应分位数0.2
        # 80%时间：对应分位数0.8 
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x + y,  # 测试的函数
            quantiles=quantiles  # 计算的分位数
        )
    
    # 测试Triton实现
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y),  # 测试的函数
            quantiles=quantiles  # 计算的分位数
        )
    
    # 计算吞吐量(GB/s)
    # 公式：3 * 数据量(元素数) * 每个元素大小(字节) / 时间(秒)
    # 3个数据流：读取x，读取y，写入结果
    def gbps(ms):
        return 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# 运行基准测试
print("运行性能基准测试...")
benchmark.run(print_data=True, show_plots=True)
print("基准测试完成！")
