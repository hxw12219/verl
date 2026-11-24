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

import pytest
import torch

from verl.utils.kernel.gather_rows import gather_rows, sum_copies


def _packed_kv_index(trunk, branches, unread=0):
    """kv_index of one shared-prefix group: the trunk, then each branch's trunk copy and its own rows."""
    rows = torch.arange(trunk + sum(branches) + unread)
    index, start = [rows[:trunk]], trunk
    for length in branches:
        index += [rows[:trunk], rows[start : start + length]]
        start += length
    return torch.cat(index), rows.numel()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gather_rows_matches_index_select(device):
    index, rows = _packed_kv_index(trunk=37, branches=[5, 9, 1, 12], unread=3)
    index = index.to(device)
    x = torch.randn(1, rows, 4, 128, device=device, dtype=torch.float32, requires_grad=True)
    ref = x.detach().clone().requires_grad_(True)
    grad = torch.randn(1, index.numel(), 4, 128, device=device)

    out = gather_rows(x, index)
    out.backward(grad)
    ref_out = ref.index_select(1, index)
    ref_out.backward(grad)

    assert torch.equal(out, ref_out)
    torch.testing.assert_close(x.grad, ref.grad)
    assert x.grad[0, -3:].eq(0).all()


def test_sum_copies_is_deterministic_and_no_less_accurate_than_index_add():
    index, rows = _packed_kv_index(trunk=700, branches=[60] * 16, unread=1)
    index = index.cuda()
    grad = torch.randn(index.numel(), 1024, device="cuda", dtype=torch.bfloat16)
    exact = torch.zeros(rows, 1024, device="cuda", dtype=torch.float64).index_add_(0, index, grad.double())

    out = sum_copies(grad, index, rows)
    atomic = torch.zeros(rows, 1024, device="cuda", dtype=torch.bfloat16).index_add_(0, index, grad)

    assert torch.equal(out, sum_copies(grad, index, rows))
    # fp32 accumulation, then one rounding to bf16 (2**-8 relative).
    torch.testing.assert_close(out.double(), exact, rtol=2**-8, atol=1e-5)
    assert (out.double() - exact).abs().max() <= (atomic.double() - exact).abs().max() + 1e-6


def test_gather_rows_rejects_batched_input():
    with pytest.raises(ValueError, match="rows"):
        gather_rows(torch.zeros(2, 3, 4), torch.tensor([0, 1]))
