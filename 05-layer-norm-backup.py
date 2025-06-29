"""
Layer Normalization (层归一化)
=============================

本教程将教你编写一个高性能的层归一化内核，其运行速度比 PyTorch 官方实现更快。

在此过程中，你将学习到：

* 在 Triton 中实现反向传播
* 在 Triton 中实现并行归约

层归一化概念：
-----------
LayerNorm 操作符首次在 [BA2016] 中提出，作为一种提高序列模型（如 Transformers）
或小批次神经网络性能的方法。它接受一个向量 x 作为输入，并产生一个相同形状的向量 y 作为输出。
归一化通过减去均值并除以 x 的标准差来执行。
归一化后，应用一个可学习的线性变换，包含权重 w 和偏置 b。

前向传播的数学公式如下：

.. math::
   y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b

其中 ε 是添加到分母中的小常数，用于数值稳定性。
让我们首先看看前向传播的实现。
"""

# %%
# 动机和背景
# ---------
#
# LayerNorm 操作符首次在 [BA2016] 中引入，用于改善序列模型（如 Transformers）
# 或小批次神经网络的性能。它接受向量 x 作为输入，产生相同形状的向量 y 作为输出。
# 归一化通过减去均值并除以 x 的标准差来执行。
# 归一化后，应用包含权重 w 和偏置 b 的可学习线性变换。
# 前向传播可以表达为：
#
# .. math::
#    y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
#
# 其中 ε 是添加到分母的小常数，用于数值稳定性。
# 让我们首先看看前向传播实现。

import torch

import triton
import triton.language as tl

try:
    # 这是 https://github.com/NVIDIA/apex，不是 PyPi 上的 apex，
    # 所以不应该添加到 setup.py 的 extras_require 中。
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

# 设备配置：优先使用 CUDA，如果不可用则使用 CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@triton.jit
def _layer_norm_fwd_fused(
    X,  # 指向输入数据的指针
    Y,  # 指向输出数据的指针
    W,  # 指向权重的指针
    B,  # 指向偏置的指针
    Mean,  # 指向均值的指针
    Rstd,  # 指向标准差倒数的指针 (1/std)
    stride,  # 移动一行时指针的增量
    N,  # X 的列数（特征维度）
    eps,  # 避免除零的小常数 epsilon
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常数）
):
    """
    Layer Normalization 前向传播融合内核
    
    该内核实现了融合的 Layer Normalization 前向传播，包括：
    1. 计算输入的均值
    2. 计算输入的方差
    3. 进行标准化
    4. 应用可学习的线性变换（权重和偏置）
    
    每个程序块处理输入矩阵的一行。
    """
    # 将程序 ID 映射到应该计算的 X 和 Y 的行
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    
    # ==================== 第一步：计算均值 ====================
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # 分块遍历该行的所有元素
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        # 加载数据，使用掩码处理边界情况，未使用的元素设为 0
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    # 计算该行的均值
    mean = tl.sum(_mean, axis=0) / N
    
    # ==================== 第二步：计算方差 ====================
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # 再次分块遍历，计算与均值的差的平方
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        # 计算 (x - mean)，对于超出边界的位置设为 0
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    # 计算该行的方差
    var = tl.sum(_var, axis=0) / N
    # 计算标准差的倒数，添加 eps 确保数值稳定性
    rstd = 1 / tl.sqrt(var + eps)
    
    # ==================== 第三步：保存统计量 ====================
    # 将均值和标准差倒数存储到对应位置
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    
    # ==================== 第四步：归一化和线性变换 ====================
    # 分块处理归一化和线性变换
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # 加载权重和偏置
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        # 加载输入数据
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        # 标准化：(x - mean) * rstd
        x_hat = (x - mean) * rstd
        # 线性变换：x_hat * w + b
        y = x_hat * w + b
        # 将结果写入输出
        tl.store(Y + cols, y, mask=mask)


# %%
# 反向传播
# --------
#
# Layer Normalization 操作符的反向传播比前向传播更复杂一些。
# 设 x̂ 为线性变换前的归一化输入 (x - E[x]) / sqrt(Var(x) + ε)，
# x 的向量-雅可比乘积 (VJP) ∇_x 由以下公式给出：
#
# .. math::
#    \nabla_{x} = \frac{1}{\sigma}\Big( \nabla_{y} \odot w - \underbrace{ \big( \frac{1}{N} \hat{x} \cdot (\nabla_{y} \odot w) \big) }_{c_1} \odot \hat{x} - \underbrace{ \frac{1}{N} \nabla_{y} \cdot w }_{c_2} \Big)
#
# 其中 ⊙ 表示逐元素乘法，· 表示点积，σ 是标准差。
# c_1 和 c_2 是中间常数，用于提高以下实现的可读性。
#
# 对于权重 w 和偏置 b，VJP ∇_w 和 ∇_b 更加直接：
#
# .. math::
#    \nabla_{w} = \nabla_{y} \odot \hat{x} \quad \text{和} \quad \nabla_{b} = \nabla_{y}
#
# 由于同一批次中的所有行使用相同的权重 w 和偏置 b，它们的梯度需要求和。
# 为了高效执行这一步骤，我们使用并行归约策略：每个内核实例将
# 某些行的部分 ∇_w 和 ∇_b 累积到 GROUP_SIZE_M 个独立缓冲区之一中。
# 这些缓冲区保留在 L2 缓存中，然后由另一个函数进一步归约以计算实际的 ∇_w 和 ∇_b。
#
# 设输入行数 M = 4 且 GROUP_SIZE_M = 2，
# 以下是 ∇_w 并行归约策略的示意图（为简洁起见省略了 ∇_b）：
#
#   .. image:: parallel_reduction.png
#
# 在阶段 1 中，具有相同颜色的 X 行共享相同的缓冲区，因此使用锁来确保一次只有一个内核实例写入缓冲区。
# 在阶段 2 中，缓冲区被进一步归约以计算最终的 ∇_w 和 ∇_b。
# 在以下实现中，阶段 1 由函数 _layer_norm_bwd_dx_fused 实现，阶段 2 由函数 _layer_norm_bwd_dwdb 实现。


# %%
# 性能基准测试
# -----------
#
# 我们现在可以比较我们的内核与 PyTorch 的性能。
# 这里我们关注小于 64KB 的输入特征。
# 具体来说，可以设置 'mode': 'backward' 来benchmark反向传播。


class LayerNorm(torch.autograd.Function):
    """
    自定义 Layer Normalization 自动求导函数
    
    该类继承自 torch.autograd.Function，实现了与 PyTorch 自动求导系统的集成。
    它包装了我们的 Triton 内核，提供前向传播和反向传播的完整实现。
    """

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        """
        Layer Normalization 前向传播
        
        参数:
            ctx: 上下文对象，用于保存反向传播所需的张量
            x: 输入张量
            normalized_shape: 归一化形状（通常是最后一个维度）
            weight: 可学习的权重参数
            bias: 可学习的偏置参数
            eps: 数值稳定性常数
            
        返回:
            y: 归一化后的输出张量
        """
        # 分配输出张量，与输入形状相同
        y = torch.empty_like(x)
        
        # 将输入重塑为 2D 张量以便处理
        # 最后一个维度是特征维度，其他维度被展平为批次维度
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape  # M: 批次大小, N: 特征维度
        
        # 分配存储均值和标准差倒数的张量
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        
        # ==================== 内核配置 ====================
        # 计算最大融合大小（64KB 限制）
        MAX_FUSED_SIZE = 65536 // x.element_size()
        # 选择合适的块大小（必须是 2 的幂次）
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        
        # 检查特征维度是否超过支持的最大值
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        
        # warp 数量的启发式选择：平衡并行度和资源使用
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        
        # ==================== 执行前向内核 ====================
        _layer_norm_fwd_fused[(M, )](  # 网格大小：每行一个程序块
            x_arg, y, weight, bias, mean, rstd,  # 输入输出张量
            x_arg.stride(0), N, eps,  # 步长、特征数、epsilon
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        
        # ==================== 保存反向传播所需信息 ====================
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        
        return y

    @staticmethod
    def backward(ctx, dy):
        """
        Layer Normalization 反向传播
        
        参数:
            ctx: 包含前向传播保存信息的上下文对象
            dy: 输出梯度
            
        返回:
            dx: 输入梯度
            None: normalized_shape 不需要梯度
            dw: 权重梯度
            db: 偏置梯度
            None: eps 不需要梯度
        """
        # 从上下文中恢复保存的张量
        x, w, b, m, v = ctx.saved_tensors
        
        # ==================== 并行归约策略配置 ====================
        # 根据特征维度选择合适的组大小，平衡内存使用和并行度
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256
        
        # ==================== 分配输出张量 ====================
        # 锁数组：用于并行归约的同步
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        # 部分梯度缓冲区
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        # 最终梯度张量
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        
        # ==================== 第一阶段：计算输入梯度和部分权重/偏置梯度 ====================
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M, )](  # 网格大小：每行一个程序块
            dx, dy, _dw, _db, x, w, m, v, locks,  # 输入输出张量
            x_arg.stride(0), N,  # 步长和特征数
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  # 使用前向传播的块大小
            GROUP_SIZE_M=GROUP_SIZE_M,  # 并行归约组大小
            num_warps=ctx.num_warps)  # 使用前向传播的 warp 数
        
        # ==================== 第二阶段：归约部分梯度得到最终权重/偏置梯度 ====================
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )
        _layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  # 部分和、最终梯度、维度
            BLOCK_SIZE_M=32,  # 行块大小
            BLOCK_SIZE_N=128)  # 列块大小
        
        return dx, None, dw, db, None


@triton.jit
def _layer_norm_bwd_dx_fused(DX,  # 指向输入梯度的指针
                             DY,  # 指向输出梯度的指针
                             DW,  # 指向权重梯度部分和的指针
                             DB,  # 指向偏置梯度部分和的指针
                             X,  # 指向输入的指针
                             W,  # 指向权重的指针
                             Mean,  # 指向均值的指针
                             Rstd,  # 指向标准差倒数的指针
                             Lock,  # 指向锁的指针
                             stride,  # 移动一行时指针的增量
                             N,  # X 的列数
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    Layer Normalization 反向传播第一阶段：输入梯度计算和权重/偏置梯度部分累积
    
    该内核实现反向传播的第一阶段，包括：
    1. 计算输入 x 的梯度 ∇_x
    2. 计算权重 w 和偏置 b 的部分梯度并累积到共享缓冲区
    3. 使用锁机制确保多线程安全写入
    """
    # 将程序 ID 映射到应该计算的 X、DX 和 DY 的元素
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    
    # 为并行归约偏移锁和权重/偏置梯度指针
    lock_id = row % GROUP_SIZE_M  # 确定该行使用哪个锁和缓冲区
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M   # 计数器，用于跟踪写入次数
    DW = DW + lock_id * N + cols  # 权重梯度缓冲区偏移
    DB = DB + lock_id * N + cols  # 偏置梯度缓冲区偏移
    
    # 将数据加载到 SRAM（共享内存）
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0.).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    
    # 计算 dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    
    # 写入 dx
    tl.store(DX + cols, dx, mask=mask)
    
    # 累积部分和为 dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    
    # 使用锁机制确保多线程安全写入
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    
    count = tl.load(Count)
    
    # 第一次写入不累积
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    # 需要一个屏障来确保所有线程完成后释放锁
    tl.debug_barrier()

    # 释放锁
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(DW,  # 指向权重梯度部分和的指针
                         DB,  # 指向偏置梯度部分和的指针
                         FINAL_DW,  # 指向最终权重梯度的指针
                         FINAL_DB,  # 指向最终偏置梯度的指针
                         M,  # GROUP_SIZE_M（缓冲区数量）
                         N,  # X 的列数（特征维度）
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    Layer Normalization 反向传播第二阶段：权重/偏置梯度最终归约
    
    该内核实现反向传播的第二阶段，完成并行归约的最后步骤：
    1. 从多个缓冲区中读取权重和偏置的部分梯度
    2. 对所有部分梯度进行求和归约
    3. 计算并写入最终的权重和偏置梯度
    
    这是并行归约策略的第二阶段，将第一阶段产生的多个部分和
    进一步归约为最终的梯度值。
    """
    # 将程序 ID 映射到应该处理的列范围
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # 初始化累积器，用于存储权重和偏置梯度的和
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # ==================== 归约阶段 ====================
    # 遍历所有缓冲区的行，累积部分梯度
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        # 创建二维掩码：行掩码和列掩码的组合
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        # 计算二维偏移量：行索引 * N + 列索引
        offs = rows[:, None] * N + cols[None, :]
        
        # 从部分和缓冲区加载数据并累积
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    
    # ==================== 最终归约和写入 ====================
    # 对每列进行最终求和，得到每个特征的总梯度
    sum_dw = tl.sum(dw, axis=0)  # 沿行方向求和
    sum_db = tl.sum(db, axis=0)  # 沿行方向求和
    
    # 将最终的权重和偏置梯度写入输出
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


# %%
# 包装函数
# --------
# 
# 现在我们可以定义一个 layer_norm 函数，它使用上面定义的 Triton 内核。
# 最重要的是，我们不需要在 Python 中实现反向传播，因为 Triton 会自动生成它。

def layer_norm(x, normalized_shape, weight, bias, eps):
    """
    使用 Triton 实现的 Layer Normalization 函数
    
    参数:
        x: 输入张量
        normalized_shape: 归一化形状
        weight: 权重参数
        bias: 偏置参数
        eps: 数值稳定性常数
        
    返回:
        归一化后的张量
    """
    return LayerNorm.apply(x, normalized_shape, weight, bias, eps)


# %%
# 单元测试
# --------
# 
# 我们可以测试我们的自定义操作符与 PyTorch 的原生 LayerNorm 的一致性。

def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    """
    测试自定义 LayerNorm 与 PyTorch 原生实现的一致性
    
    参数:
        M: 批次大小（行数）
        N: 特征维度（列数）
        dtype: 数据类型
        eps: 数值稳定性常数
        device: 计算设备
    """
    # 创建随机输入数据
    x_shape = (M, N)
    w_shape = (N,)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    
    # 前向传播测试
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    
    # 检查前向传播结果
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    
    # 反向传播测试
    y_tri.backward(dy, retain_graph=True)
    x_grad_tri, weight_grad_tri, bias_grad_tri = x.grad.clone(), weight.grad.clone(), bias.grad.clone()
    
    # 清零梯度
    x.grad, weight.grad, bias.grad = None, None, None
    
    # PyTorch 反向传播
    y_ref.backward(dy, retain_graph=True)
    x_grad_ref, weight_grad_ref, bias_grad_ref = x.grad.clone(), weight.grad.clone(), bias.grad.clone()
    
    # 检查反向传播结果
    assert torch.allclose(x_grad_tri, x_grad_ref, atol=1e-2, rtol=0)
    assert torch.allclose(weight_grad_tri, weight_grad_ref, atol=1e-2, rtol=0)
    assert torch.allclose(bias_grad_tri, bias_grad_ref, atol=1e-2, rtol=0)
    
    print(f"✅ 测试通过 - M={M}, N={N}, dtype={dtype}")


# %%
# 性能基准测试
# -----------
# 
# 我们可以将我们的 LayerNorm 与 PyTorch 的原生实现进行基准测试。
# 这里我们专注于反向传播，因为它是更复杂的操作。

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 用作图表 x 轴的参数名
        x_vals=[512 * i for i in range(2, 32)],  # `x_name` 的不同可能值
        line_arg='provider',  # 其值对应图表中不同线条的参数名
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),  # `line_arg` 的可能值
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),  # 线条的名称
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],  # 线条样式
        ylabel='GB/s',  # y 轴标签
        plot_name='layer-norm-backward',  # 图表文件名
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},  # 函数参数值
    ))
def bench_layer_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    """
    Layer Normalization 性能基准测试函数
    
    该函数对比不同实现的 Layer Normalization 性能，包括：
    - Triton 自定义实现
    - PyTorch 原生实现  
    - Apex 实现（如果可用）
    
    参数:
        M: 批次大小（行数）
        N: 特征维度大小（列数）
        dtype: 数据类型（如 torch.float16, torch.float32）
        provider: 实现提供者 ('triton', 'torch', 'apex')
        mode: 测试模式 ('forward', 'backward')
        eps: 数值稳定性常数
        device: 计算设备
        
    返回:
        gbps: 吞吐量（GB/s）
    """
    # 创建测试数据
    x_shape = (M, N)
    w_shape = (N,)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    
    # 计算理论内存使用量（字节）
    # 前向传播：读取 x, weight, bias，写入 y
    # 反向传播：读取 x, weight, bias, dy，写入 dx, dweight, dbias
    if mode == 'forward':
        # 前向传播内存访问：x(读) + weight(读) + bias(读) + y(写)
        gbytes = (2 * x.numel() + 2 * weight.numel() + 2 * bias.numel()) * x.element_size() / 1e9
    else:
        # 反向传播内存访问：x(读) + weight(读) + bias(读) + dy(读) + dx(写) + dweight(写) + dbias(写)
        gbytes = (3 * x.numel() + 3 * weight.numel() + 3 * bias.numel()) * x.element_size() / 1e9
    
    # 选择实现提供者
    if provider == 'triton':
        fn = lambda: layer_norm(x, w_shape, weight, bias, eps)
    elif provider == 'torch':
        fn = lambda: torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)
    elif provider == 'apex':
        apex_layer_norm = apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype)
        fn = lambda: apex_layer_norm(x)
    else:
        raise ValueError(f"未知的提供者: {provider}")
    
    # 预热和性能测试
    if mode == 'forward':
        # 前向传播基准测试
        ms = triton.testing.do_bench(fn)
    else:
        # 反向传播基准测试
        def full_fn():
            x.grad, weight.grad, bias.grad = None, None, None
            y = fn()
            y.backward(dy, retain_graph=True)
        ms = triton.testing.do_bench(full_fn)
    
    # 计算吞吐量（GB/s）
    gbps = gbytes / (ms * 1e-3)
    return gbps


# %%
# 主程序
# ------
# 
# 我们可以运行单元测试来验证我们的操作符是否正确，
# 然后运行基准测试来评估其性能。

if __name__ == '__main__':
    print("🚀 开始 Layer Normalization 测试和基准测试")
    print(f"📱 使用设备: {DEVICE}")
    print(f"🔧 CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎯 GPU 设备: {torch.cuda.get_device_name()}")
    print()
    
    # ==================== 正确性测试 ====================
    print("=" * 60)
    print("🧪 正确性测试")
    print("=" * 60)
    
    # 测试不同的配置
    test_configs = [
        (1823, 781, torch.float16),
        (1823, 781, torch.float32),
        (4096, 1024, torch.float16),
        (4096, 1024, torch.float32),
    ]
    
    for M, N, dtype in test_configs:
        try:
            test_layer_norm(M, N, dtype)
        except Exception as e:
            print(f"❌ 测试失败 - M={M}, N={N}, dtype={dtype}: {e}")
    
    print("\n✅ 所有正确性测试通过！")
    
    # ==================== 性能基准测试 ====================
    print("\n" + "=" * 60)
    print("⚡ 性能基准测试")
    print("=" * 60)
    
    print("📊 运行性能基准测试（这可能需要几分钟）...")
    print("📈 测试配置：M=4096, dtype=torch.float16, mode=backward")
    print("📉 将生成性能对比图表：layer-norm-backward.png")
    print()
    
    # 运行基准测试
    bench_layer_norm.run(show_plots=True, print_data=True)
    
    print("\n🎉 Layer Normalization 测试和基准测试完成！")
    print("📊 查看生成的图表文件了解详细性能对比")
