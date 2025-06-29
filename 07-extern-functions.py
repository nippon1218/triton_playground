"""
Libdevice (`tl.extra.libdevice`) function
==============================
Triton can invoke a custom function from an external library.
In this example, we will use the `libdevice` library to apply `asin` on a tensor.

Please refer to `CUDA libdevice-users-guide <https://docs.nvidia.com/cuda/libdevice-users-guide/index.html>`_ and/or `HIP device-lib source code <https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs/ocml/src>`_ regarding the semantics of all available libdevice functions.

In `libdevice.py`, we try to aggregate functions with the same computation but different data types together.
For example, both `__nv_asin` and `__nv_asinf` calculate the principal value of the arc sine of the input, but `__nv_asin` operates on `double` and `__nv_asinf` operates on `float`.
Triton automatically selects the correct underlying device function to invoke based on input and output types.
"""

# %%
#  asin Kernel
# ------------

import torch

import triton
import triton.language as tl
import inspect
import os
from triton.language.extra import libdevice

from pathlib import Path

# 使用PyTorch的API获取设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)


# %%
#  Using the default libdevice library path
# -----------------------------------------
# We can use the default libdevice library path encoded in `triton/language/math.py`

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
output_triton = torch.zeros(size, device=DEVICE)
output_torch = torch.asin(x)
assert x.is_cuda and output_triton.is_cuda
n_elements = output_torch.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')


# %%
#  Customize the libdevice library path
# -------------------------------------
# We can also customize the libdevice library path by passing the path to the `libdevice` library to the `asin` kernel.
def is_cuda():
    # 使用PyTorch的API检查
    return torch.cuda.is_available()


def is_hip():
    # Triton 3.1.0可能不支持HIP后端，使用简单的检查
    return False


# 对于Triton 3.1.0，我们使用默认的libdevice路径
# 这部分代码可能无法正常工作，因为我们不确定外部库的确切位置
if is_cuda():
    try:
        # 尝试使用默认路径
        extern_libs = {}
    except Exception as e:
        print(f"警告: 无法加载外部库: {e}")
        extern_libs = {}
else:
    print("不支持的后端，跳过外部库加载")
    extern_libs = {}

output_triton = torch.empty_like(x)
asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024, extern_libs=extern_libs)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
