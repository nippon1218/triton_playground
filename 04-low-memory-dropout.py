"""
中文说明：
这个教程展示了如何使用Triton实现低内存消耗的Dropout操作。
传统的Dropout实现通常需要存储一个与输入相同大小的掩码张量，
而这里我们将展示如何仅使用一个int32种子来实现相同功能，
从而大大减少内存使用并简化随机状态管理。

Low-Memory Dropout
==================

在这个教程中，您将编写一个内存高效的Dropout实现，其状态由单个int32种子组成。
这与传统的Dropout实现不同，传统的Dropout实现的状态通常由与输入相同形状的位掩码张量组成。

通过这样做，您将了解：

* 使用PyTorch实现Dropout的局限性。

* 在Triton中进行并行伪随机数生成。

"""


# %%
# Baseline
# --------
#
# Dropout操作最初是在 [SRIVASTAVA2014]_ 中引入的，作为一种提高深度神经网络在低数据环境下的性能的方法（即正则化）。
# The *dropout* operator was first introduced in [SRIVASTAVA2014]_ as a way to improve the performance
# of deep neural networks in low-data regime (i.e. regularization).
#
# Dropout接收一个向量作为输入，并产生相同形状的输出向量。输出中的每个标量有概率 :math:`p` 被置为零，
# 否则它将从输入中复制。这迫使网络即使只有 :math:`1 - p` 比例的输入标量可用时也能表现良好。
#
# 在评估阶段，我们希望使用网络的全部能力，所以设置 :math:`p=0`。但这样会增加输出的范数
# （这可能是不好的，例如，它可能导致输出softmax温度人为降低）。为了防止这种情况，
# 我们将输出乘以 :math:`\frac{1}{1 - p}`，这样无论dropout概率如何，都能保持范数一致。
#
# Let's first take a look at the baseline implementation.

import tabulate
import torch

import triton
import triton.language as tl

# 使用直接的设备分配替换有问题的行
# 检查是否有可用的CUDA设备，如果有则使用GPU，否则使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@triton.jit
def _dropout(
    x_ptr,  # 指向输入的指针
    x_keep_ptr,  # 指向由0和1组成的掩码的指针
    output_ptr,  # 指向输出的指针
    n_elements,  # `x`张量中的元素数量
    p,  # `x`中元素被置为零的概率
    BLOCK_SIZE: tl.constexpr,  # 编译时常量，表示每个线程块处理的元素数量
):
    # 获取当前程序（线程块）的ID
    pid = tl.program_id(axis=0)
    # 计算当前块的起始偏移量
    block_start = pid * BLOCK_SIZE
    # 计算当前块内每个线程的偏移量
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建掩码，确保我们不会越界访问
    mask = offsets < n_elements
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)  # 加载输入数据
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)  # 加载保留掩码
    # 下面这行是关键部分，如上文所述！
    # 如果x_keep为真，则保留输入值（除以(1-p)进行缩放），否则置为0
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # 将结果写回内存
    tl.store(output_ptr + offsets, output, mask=mask)


# Dropout的Python包装函数
def dropout(x, x_keep, p):
    # 创建一个与输入相同形状的空张量用于存储结果
    output = torch.empty_like(x)
    # 确保输入张量是连续的内存布局
    assert x.is_contiguous()
    # 计算输入张量中的元素总数
    n_elements = x.numel()
    # 定义计算网格大小的函数，确保覆盖所有元素
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # 调用Triton内核，BLOCK_SIZE设为1024
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


# Input tensor
x = torch.randn(size=(10, ), device=DEVICE)
# Dropout mask
p = 0.5
x_keep = (torch.rand(size=(10, ), device=DEVICE) > p).to(torch.int32)
#
output = dropout(x, x_keep=x_keep, p=p)
print(tabulate.tabulate([
    ["input"] + x.tolist(),
    ["keep mask"] + x_keep.tolist(),
    ["output"] + output.tolist(),
]))

# %%
# 基于种子的Dropout
# --------------
#
# 上面的Dropout实现虽然可以工作，但使用起来有些不便。首先，我们需要存储dropout掩码以便反向传播。
# 其次，当使用重计算/检查点时，dropout状态管理会变得非常棘手（例如，参见
# https://pytorch.org/docs/stable/checkpoint.html 中关于`preserve_rng_state`的所有注释）。
# 在本教程中，我们将描述一种替代实现，它：
# (1) 具有更小的内存占用；
# (2) 需要更少的数据移动；
# (3) 简化了在多次内核调用中保持随机性的管理。
#
# Triton中的伪随机数生成非常简单！在本教程中，我们将使用
# :code:`triton.language.rand` 函数，它接收一个种子和一个 :code:`int32` 偏移量块，
# 生成一个均匀分布在[0, 1)范围内的 :code:`float32` 值块。
# 如果需要，Triton还提供其他 :ref:`随机数生成策略<Random Number Generation>`。
#
# .. note::
#    Triton's implementation of PRNG is based on the Philox algorithm (described on [SALMON2011]_).
#
# Let's put it all together.


@triton.jit
def _seeded_dropout(
    x_ptr,      # 指向输入的指针
    output_ptr, # 指向输出的指针
    n_elements, # 元素数量
    p,          # dropout概率
    seed,       # 随机数生成的种子
    BLOCK_SIZE: tl.constexpr,  # 每个线程块处理的元素数量
):
    # 计算此实例处理的元素的内存偏移量
    pid = tl.program_id(axis=0)  # 获取当前程序ID
    block_start = pid * BLOCK_SIZE  # 计算块起始位置
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 计算每个线程的偏移量
    # 从x加载数据
    mask = offsets < n_elements  # 创建边界掩码
    x = tl.load(x_ptr + offsets, mask=mask)  # 加载输入数据
    # 随机剪枝
    random = tl.rand(seed, offsets)  # 使用种子和偏移量生成随机数
    x_keep = random > p  # 创建dropout掩码
    # 写回结果
    output = tl.where(x_keep, x / (1 - p), 0.0)  # 应用dropout并缩放
    tl.store(output_ptr + offsets, output, mask=mask)  # 存储结果


# 基于种子的Dropout的Python包装函数
def seeded_dropout(x, p, seed):
    # 创建输出张量
    output = torch.empty_like(x)
    # 确保输入是连续的
    assert x.is_contiguous()
    # 计算元素总数
    n_elements = x.numel()
    # 定义计算网格
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # 调用Triton内核
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


x = torch.randn(size=(10, ), device=DEVICE)
# 与基准相比 - 这里从不实例化dropout掩码！这是内存优化的关键
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))

# %%
# 太好了！我们有了一个Triton内核，只要种子相同，它就会应用相同的dropout掩码！
# 如果您想进一步探索GPU编程中伪随机性的应用，我们鼓励您
# 探索`python/triton/language/random.py`！

# %%
# 练习
# ---------
#
# 1. 扩展内核以处理矩阵，并使用种子向量 - 每行一个种子。
# 2. 添加对步长(striding)的支持。
# 3. (挑战) 实现一个稀疏Johnson-Lindenstrauss变换的内核，每次使用种子动态生成投影矩阵。

# %%
# References
# ----------
#
# .. [SALMON2011] John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", 2011
# .. [SRIVASTAVA2014] Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014
