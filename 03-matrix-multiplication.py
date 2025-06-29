"""
矩阵乘法实现
=====================
本教程将指导您编写一个高性能的FP16矩阵乘法内核，其性能可媲美cuBLAS或rocBLAS。

您将具体学习以下内容：

* 块级矩阵乘法实现
* 多维指针算术运算
* 程序重排序以提高L2缓存命中率
* 自动性能调优
"""

# %%
# 动机
# -----------
#
# 矩阵乘法是现代高性能计算系统中的关键构建块。
# 由于优化难度大，通常由硬件供应商自己实现，作为所谓的"内核库"（如cuBLAS）的一部分。
# 不幸的是，这些库通常是专有的，难以根据现代深度学习工作负载的需求（如融合激活函数）进行定制。
# 本教程将展示如何使用Triton实现高效的矩阵乘法，这种方式既易于定制又易于扩展。
#
# 简单来说，我们将编写的内核将实现以下分块算法来将(M,K)矩阵与(K,N)矩阵相乘：
#
#  .. code-block:: python
#
#    # 并行执行
#    for m in range(0, M, BLOCK_SIZE_M):
#      # 并行执行
#      for n in range(0, N, BLOCK_SIZE_N):
#        # 初始化累加器
#        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#        # 计算内积
#        for k in range(0, K, BLOCK_SIZE_K):
#          # 加载A的子矩阵块
#          a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#          # 加载B的子矩阵块
#          b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#          # 矩阵乘法累加
#          acc += dot(a, b)
#        # 将结果写回C
#        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
#
# 其中，双重嵌套for循环的每次迭代都由一个专用的Triton程序实例执行。

# %%
# 计算内核
# --------------
# --------------
#
# 上述算法在Triton中实现起来实际上相当直接。
# 主要的难点在于计算内层循环中需要读取的A和B块的内存位置。
# 为此，我们需要使用多维指针算术。
#
# 指针算术
# ~~~~~~~~~~~~~~~~~~~
#
# 对于行主序的2D张量 :code:`X`，:code:`X[i, j]` 的内存位置由
# :code:`&X[i, j] = X + i*stride_xi + j*stride_xj` 给出。
# 因此，:code:`A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]` 和
# :code:`B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]` 的指针块可以用以下伪代码定义：
#
#  .. code-block:: python
#
#    # A的子矩阵块指针计算
#    &A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);
#    # B的子矩阵块指针计算
#    &B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);
#
# 这意味着我们可以用以下代码在Triton中初始化A和B的块指针（即 :code:`k=0` 时）。
# 注意，我们需要额外的模运算来处理 :code:`M` 不是 :code:`BLOCK_SIZE_M` 的倍数或
# :code:`N` 不是 :code:`BLOCK_SIZE_N` 的倍数的情况，这时我们可以用一些无用的值填充数据，
# 这些值不会对结果产生影响。对于 :code:`K` 维度，我们稍后将使用掩码加载语义来处理。
#
#  .. code-block:: python
#
#    # 计算A的行偏移（考虑边界条件）
#    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#    # 计算B的列偏移（考虑边界条件）
#    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#    # K维度的偏移
#    offs_k = tl.arange(0, BLOCK_SIZE_K)
#    # 计算A的指针（使用广播进行高效计算）
#    a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
#    # 计算B的指针（使用广播进行高效计算）
#    b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)
#
# 然后在内部循环中按如下方式更新指针：
#
#  .. code-block:: python
#
#    # 更新A的指针到下一个K块
#    a_ptrs += BLOCK_SIZE_K * stride_ak;
#    # 更新B的指针到下一个K块
#    b_ptrs += BLOCK_SIZE_K * stride_bk;
#
#
# L2缓存优化
# ~~~~~~~~~~~~~~~~~~~~~~
#
# 如上所述，每个程序实例计算 :code:`C` 的一个 :code:`[BLOCK_SIZE_M, BLOCK_SIZE_N]` 块。
# 重要的是要记住，计算这些块的顺序确实很重要，因为它会影响我们程序的L2缓存命中率。
# 不幸的是，简单的行主序排序（如下所示）效果并不理想：
#
#  .. code-block:: Python
#
#    # 程序ID
#    pid = tl.program_id(axis=0)
#    # 计算N方向上的网格大小
#    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
#    # 计算M方向上的程序ID
#    pid_m = pid // grid_n
#    # 计算N方向上的程序ID
#    pid_n = pid % grid_n
#
# 一个可能的解决方案是以促进数据重用的顺序启动块。
# 这可以通过在切换到下一列之前，将块按 :code:`GROUP_M` 行分组（'超级分组'）来实现：
#
#  .. code-block:: python
#
#    # 程序ID
#    pid = tl.program_id(axis=0)
#    # M轴上的程序ID数量
#    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#    # N轴上的程序ID数量
#    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#    # 每个组中的程序数量
#    num_pid_in_group = GROUP_SIZE_M * num_pid_n
#    # 当前程序所在的组ID
#    group_id = pid // num_pid_in_group
#    # 组中第一个程序的行ID
#    first_pid_m = group_id * GROUP_SIZE_M
#    # 如果`num_pid_m`不能被`GROUP_SIZE_M`整除，则最后一组较小
#    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#    # *在组内*，程序按列主序排序
#    # 在*启动网格*中程序的行ID
#    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#    # 在*启动网格*中程序的列ID
#    pid_n = (pid % num_pid_in_group) // group_size_m
#
# 例如，在下述矩阵乘法中，每个矩阵是9x9个块，我们可以看到，如果按行主序计算输出，
# 我们需要将90个块加载到SRAM中以计算前9个输出块，但如果使用分组排序，
# 我们只需要加载54个块。
#
#   .. image:: grouped_vs_row_major_ordering.png
#
# 实际上，在某些硬件架构上，这可以将我们的矩阵乘法内核性能提高10%以上
#（例如，在A100上从220 TFLOPS提高到245 TFLOPS）。
#

# %%
# 最终实现
# ------------
# 导入必要的库
import torch

import triton
import triton.language as tl

# 设置设备：如果CUDA可用则使用GPU，否则使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_cuda():
    """检查当前是否可以使用CUDA"""
    return torch.cuda.is_available()


def is_hip_cdna2():
    """
    检查是否为AMD CDNA2架构的GPU
    简化版本，假设不是CDNA2设备
    """
    return False


def get_cuda_autotune_config():
    """
    获取CUDA设备的自动调优配置
    返回一个配置列表，每个配置包含不同的块大小、组大小、阶段数和warp数
    """
    return [
        # 适用于FP16输入的高性能配置
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # 适用于FP8输入的高性能配置
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    """
    获取HIP设备(AMD GPU)的自动调优配置
    返回一个配置列表，每个配置针对AMD GPU架构进行了优化
    """
    return [
        # 高性能配置，针对不同形状的矩阵乘法优化
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=2),
    ]


def get_autotune_config():
    """
    根据当前设备类型返回相应的自动调优配置
    自动检测是CUDA设备还是HIP设备，并返回对应的配置
    """
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# `triton.jit`装饰器可以与`triton.autotune`一起使用来自动调优内核，它需要：
#   - 一个`triton.Config`对象列表，定义不同的元参数（如`BLOCK_SIZE_M`）和编译选项（如`num_warps`）
#   - 一个自动调优*键*，其值的变化将触发所有配置的评估
@triton.autotune(
    configs=get_autotune_config(),  # 获取自动调优配置
    key=['M', 'N', 'K'],  # 当这些维度的值变化时，重新进行自动调优
)
@triton.jit
def matmul_kernel(
        # 矩阵指针
        a_ptr,  # 矩阵A的指针
        b_ptr,  # 矩阵B的指针
        c_ptr,  # 结果矩阵C的指针
        # 矩阵维度
        M,  # 矩阵A的行数，矩阵C的行数
        N,  # 矩阵B的列数，矩阵C的列数
        K,  # 矩阵A的列数，矩阵B的行数
        # 步长变量表示在特定维度上移动一个元素时指针需要增加的字节数
        # 例如，`stride_am`表示在A中向下移动一行需要增加`a_ptr`的字节数
        stride_am,  # A矩阵行方向的步长（每行的元素个数）
        stride_ak,  # A矩阵列方向的步长
        stride_bk,  # B矩阵行方向的步长
        stride_bn,  # B矩阵列方向的步长
        stride_cm,  # C矩阵行方向的步长
        stride_cn,  # C矩阵列方向的步长
        # 元参数（在编译时常量）
        BLOCK_SIZE_M: tl.constexpr,  # M方向的块大小
        BLOCK_SIZE_N: tl.constexpr,  # N方向的块大小
        BLOCK_SIZE_K: tl.constexpr,  # K方向的块大小
        GROUP_SIZE_M: tl.constexpr,  # 组大小（用于L2缓存优化）
        ACTIVATION: tl.constexpr  # 激活函数类型
):
    """
    计算矩阵乘法 C = A x B 的内核函数
    A的形状为(M, K)，B的形状为(K, N)，C的形状为(M, N)
    """
    # -----------------------------------------------------------
    # 将程序ID `pid` 映射到它应该计算的C的块
    # 使用分组排序以促进L2数据重用
    # 详见上面的`L2缓存优化`部分
    pid = tl.program_id(axis=0)  # 获取当前程序的一维ID
    # 计算M和N方向上的块数（向上取整）
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  # M方向上的块数
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # N方向上的块数
    # 计算每组中的程序数量
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # 计算当前程序所在的组ID
    group_id = pid // num_pid_in_group
    # 计算组中第一个程序在M方向上的ID
    first_pid_m = group_id * GROUP_SIZE_M
    # 计算当前组在M方向上的实际大小（最后一组可能较小）
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # 计算当前程序在启动网格中的行ID和列ID
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # 为A和B的第一个块创建指针
    # 当我们在K方向移动时，将更新这些指针
    # `a_ptrs` 是一个形状为 [BLOCK_SIZE_M, BLOCK_SIZE_K] 的指针块
    # `b_ptrs` 是一个形状为 [BLOCK_SIZE_K, BLOCK_SIZE_N] 的指针块
    # 详见上面的`指针算术`部分
    
    # 计算A矩阵当前块的行偏移（考虑边界条件）
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    # 计算B矩阵当前块的列偏移（考虑边界条件）
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # K维度的偏移
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 计算A矩阵当前块的指针（使用广播进行高效计算）
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # 计算B矩阵当前块的指针（使用广播进行高效计算）
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # 迭代计算C矩阵的一个块
    # 我们使用fp32累加器以提高精度
    # 循环结束后，累加器将转换回fp16
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 在K维度上进行分块计算
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载A和B的下一个块，通过检查K维度生成掩码
        # 如果超出边界，则设置为0
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # 沿K维度累加矩阵乘法结果
        accumulator = tl.dot(a, b, accumulator)
        
        # 将指针前进到下一个K块
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # 可以在累加器仍为FP32时融合任意激活函数！
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    
    # 将累加器转换为FP16以节省存储空间
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # 使用掩码将输出矩阵C的块写回
    # 计算C矩阵当前块的行和列偏移
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # 计算C矩阵当前块的指针
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # 创建掩码以处理边界条件（当M或N不是BLOCK_SIZE的倍数时）
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # 使用掩码将结果存储回C矩阵
    tl.store(c_ptrs, c, mask=c_mask)


# 我们可以通过将`leaky_relu`作为`matmul_kernel`中的`ACTIVATION`元参数来融合它
@triton.jit
def leaky_relu(x):
    """
    Leaky ReLU激活函数
    当x >= 0时返回x，否则返回0.01 * x
    """
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# 现在我们可以创建一个方便的封装函数，它只需要两个输入张量，并：
# 1. 检查形状约束
# 2. 分配输出张量
# 3. 启动上面的内核

def matmul(a, b, activation=""):
    """
    执行矩阵乘法C = A x B的便捷函数
    
    参数:
        a: 形状为(M, K)的输入张量
        b: 形状为(K, N)的输入张量
        activation: 要应用的激活函数（目前仅支持'leaky_relu'或''）
    
    返回:
        c: 形状为(M, N)的结果张量
    """
    # 检查约束条件
    assert a.shape[1] == b.shape[0], "维度不兼容"
    assert a.is_contiguous(), "矩阵A必须是连续的"
    
    # 获取矩阵维度
    M, K = a.shape
    K, N = b.shape
    
    # 分配输出张量
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # 定义1D启动网格，其中每个块运行自己的程序
    # 计算需要的块数（向上取整）
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    
    # 启动内核
    matmul_kernel[grid](
        a, b, c,  # 输入和输出张量
        M, N, K,  # 矩阵维度
        a.stride(0), a.stride(1),  # A的步长
        b.stride(0), b.stride(1),  # B的步长
        c.stride(0), c.stride(1),  # C的步长
        ACTIVATION=activation  # 激活函数类型
    )
    return c


# %%
# 单元测试
# ---------
#
# 我们可以将自定义的矩阵乘法操作与原生PyTorch实现（即cuBLAS）进行对比测试

# 设置随机种子以确保结果可重现
torch.manual_seed(0)

# 创建两个随机FP16矩阵
a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)

# 使用Triton实现计算矩阵乘法
triton_output = matmul(a, b)
# 使用PyTorch原生实现计算矩阵乘法
torch_output = torch.matmul(a, b)

# 打印结果（通常注释掉，因为输出较大）
# print(f"triton_output_with_fp16_inputs={triton_output}")
# print(f"torch_output_with_fp16_inputs={torch_output}")

# 对于AMD CDNA2设备使用更大的容差
# CDNA2设备使用降低精度的fp16和bf16，并将输入和输出的非规格化值刷新为零
# 详细信息请参阅：https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
rtol = 1e-2 if is_hip_cdna2() else 0

# 检查Triton和PyTorch的结果是否在允许的误差范围内匹配
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton和PyTorch结果匹配")
else:
    print("❌ Triton和PyTorch结果不同")

# 检查当前PyTorch版本是否支持FP8数据类型（8位浮点数，5位指数，2位尾数）
TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
# 如果支持FP8且当前设备是CUDA，则运行FP8测试
if TORCH_HAS_FP8 and is_cuda():
    # 设置随机种子以确保结果可重现
    torch.manual_seed(0)
    
    # 创建两个随机FP16矩阵
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    
    # 将输入转换为FP8格式（降低精度以节省内存和计算资源）
    a = a.to(torch.float8_e5m2)  # 转换为8位浮点数（5位指数，2位尾数）
    
    # 为了提高效率，预先转置B矩阵
    # 这是因为矩阵乘法在内存访问模式上对转置后的矩阵更友好
    b = b.T
    b = b.to(torch.float8_e5m2)  # 同样转换为8位浮点数
    
    # 使用Triton实现计算FP8矩阵乘法
    triton_output = matmul(a, b)
    
    # 为了与PyTorch实现比较，需要将FP8转回FP16
    # 注意：当前PyTorch的matmul原生不支持FP8，所以需要转换
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    
    # 打印结果（通常注释掉，因为输出较大）
    # print(f"triton_output_with_fp8_inputs={triton_output}")
    # print(f"torch_output_with_fp8_inputs={torch_output}")
    
    # 检查Triton和PyTorch的结果是否在允许的误差范围内匹配
    # 对于FP8，我们使用更大的绝对容差(0.125)，因为FP8的精度较低
    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        print("✅ Triton和PyTorch FP8结果匹配")
    else:
        print("❌ Triton和PyTorch FP8结果不同")

# %%
# 基准测试
# ---------
#
# 方阵性能测试
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 现在我们可以将我们的内核性能与cuBLAS（NVIDIA）或rocBLAS（AMD）进行比较。
# 这里我们主要关注方阵的性能测试，但你可以根据需要修改脚本来测试其他形状的矩阵。

# 根据当前设备选择参考库：CUDA设备使用cuBLAS，AMD设备使用rocBLAS
ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

# 准备基准测试配置
configs = []
# 测试FP16和FP8两种输入类型（如果支持FP8）
for fp8_inputs in [False, True]:
    # 如果测试FP8但不支持FP8或不是CUDA设备，则跳过
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    
    # 添加基准测试配置
    configs.append(
        triton.testing.Benchmark(
            # 绘图时用作x轴的参数名（M, N, K）
            x_names=["M", "N", "K"],
            # x轴的不同可能值：从256开始，以128为步长，直到4096
            x_vals=[128 * i for i in range(2, 33)],
            # 用于区分图中不同线的参数名
            line_arg="provider",
            # 对于FP8情况，只比较Triton实现（因为PyTorch原生不支持FP8矩阵乘法）
            # 对于FP16情况，比较参考库（cuBLAS/rocBLAS）和Triton实现
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],
            # 图中线的标签
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],
            # 线型样式：绿色实线用于参考库，蓝色实线用于Triton
            styles=[("green", "-"), ("blue", "-")],
            # y轴标签：TFLOPS（每秒万亿次浮点运算）
            ylabel="TFLOPS",
            # 图表名称，也用作保存图表的文件名
            plot_name="matmul-performance-" + ("fp16" if not fp8_inputs else "fp8"),
            # 传递给基准测试函数的其他参数
            args={"fp8_inputs": fp8_inputs},
        ))


# 使用@triton.testing.perf_report装饰器定义基准测试函数
# 该装饰器会为configs中的每个配置生成一个测试用例
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    """
    执行矩阵乘法的基准测试函数
    
    参数:
        M, N, K: 输入矩阵的维度，形状为(M, K)的矩阵A与形状为(K, N)的矩阵B相乘
        provider: 提供矩阵乘法的实现，'triton'或参考库(cuBLAS/rocBLAS)
        fp8_inputs: 布尔值，指示是否使用FP8输入
        
    返回:
        (平均TFLOPS, 最小TFLOPS, 最大TFLOPS)
    """
    # 创建随机输入矩阵，使用FP16精度
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    
    # 如果测试FP8且支持FP8，则将输入转换为FP8格式
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)  # 转换为8位浮点数（5位指数，2位尾数）
        b = b.T  # 转置B矩阵以提高内存访问效率
        b = b.to(torch.float8_e5m2)  # 转换为8位浮点数
    
    # 定义性能分位数：中位数(50%)、低分位数(20%)和高分位数(80%)
    quantiles = [0.5, 0.2, 0.8]
    
    # 根据provider选择要测试的实现
    if provider == ref_lib.lower():
        # 使用参考库(cuBLAS/rocBLAS)执行矩阵乘法并测量性能
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b),  # 使用PyTorch的矩阵乘法（底层调用cuBLAS/rocBLAS）
            quantiles=quantiles  # 指定要计算的分位数
        )
    if provider == 'triton':
        # 使用Triton实现执行矩阵乘法并测量性能
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b),  # 使用自定义的Triton矩阵乘法
            quantiles=quantiles  # 指定要计算的分位数
        )
    
    # 计算性能指标（TFLOPS：每秒万亿次浮点运算）
    # 矩阵乘法的计算量为2*M*N*K（每个输出元素需要K次乘加运算，每次乘加包含一次乘法和一次加法）
    # 1e-12将浮点运算转换为万亿次，1e-3将毫秒转换为秒
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    
    # 返回平均、最小和最大TFLOPS
    return perf(ms), perf(max_ms), perf(min_ms)


# 运行基准测试并显示结果
# show_plots=True: 显示性能图表
# print_data=True: 打印详细的性能数据
benchmark.run(show_plots=True, print_data=True)
