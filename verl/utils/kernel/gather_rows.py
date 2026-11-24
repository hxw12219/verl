# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Row gather whose backward sums duplicate rows without atomics.

``x.index_select(1, index)`` backpropagates through a bf16 ``index_add_``. When ``index`` repeats rows many times, as
shared-prefix packing does for every trunk row, the atomics on those rows serialize. ``gather_rows`` sorts the index in
its backward and sums each row's copies in fp32 in one Triton program per row, which is faster, deterministic and
closer to the exact sum.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

__all__ = ["gather_rows", "sum_copies"]

_BLOCK = 512

if HAVE_TRITON:

    @triton.jit
    def _sum_copies_kernel(grad_ptr, order_ptr, row_ptr, out_ptr, cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < cols
        lo = tl.load(row_ptr + row)
        hi = tl.load(row_ptr + row + 1)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(lo, hi):
            src = tl.load(order_ptr + k).to(tl.int64)
            acc += tl.load(grad_ptr + src * cols + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row.to(tl.int64) * cols + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def sum_copies(grad: torch.Tensor, index: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Return ``out`` of shape ``(num_rows, C)`` with ``out[r] = sum(grad[k] for k where index[k] == r)``.

    Args:
        grad: ``(len(index), C)`` gradient of the gathered rows.
        index: Row id of each gathered row.
        num_rows: Rows of the gather's input.
    """
    if grad.dim() != 2 or grad.shape[0] != index.numel():
        raise ValueError(f"expected a ({index.numel()}, C) gradient, got {tuple(grad.shape)}")
    if not HAVE_TRITON or not grad.is_cuda:
        return torch.zeros(num_rows, grad.shape[1], dtype=grad.dtype, device=grad.device).index_add_(0, index, grad)
    grad = grad.contiguous()
    rows, order = torch.sort(index, stable=True)
    # Row r's copies are order[row_ptr[r]:row_ptr[r + 1]]; searchsorted, unlike bincount, needs no host sync.
    row_ptr = torch.searchsorted(rows, torch.arange(num_rows + 1, device=index.device, dtype=rows.dtype))
    out = torch.empty(num_rows, grad.shape[1], dtype=grad.dtype, device=grad.device)
    if num_rows:
        grid = (num_rows, triton.cdiv(grad.shape[1], _BLOCK))
        _sum_copies_kernel[grid](grad, order, row_ptr, out, grad.shape[1], BLOCK=_BLOCK)
    return out


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(index)
        ctx.x_shape = x.shape
        return x.index_select(1, index)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (index,) = ctx.saved_tensors
        out = sum_copies(grad.reshape(index.numel(), -1), index, ctx.x_shape[1])
        return out.reshape(ctx.x_shape), None


def gather_rows(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``x.index_select(1, index)`` for a ``(1, rows, ...)`` tensor, with an atomic-free backward."""
    if x.dim() < 2 or x.shape[0] != 1:
        raise ValueError(f"expected a (1, rows, ...) tensor, got {tuple(x.shape)}")
    return _GatherRows.apply(x, index)
