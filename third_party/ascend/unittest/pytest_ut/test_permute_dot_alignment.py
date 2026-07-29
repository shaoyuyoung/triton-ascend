# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def permute_dot_alignment_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    indices = offsets[:, None] * BLOCK + offsets[None, :]
    x = tl.load(x_ptr + indices)
    y = tl.load(y_ptr + indices)
    merged = tl.maximum(x, y)
    transposed = tl.permute(merged, (1, 0))
    limited = tl.minimum(y, transposed)
    output = tl.dot(x, limited, input_precision="ieee", out_dtype=tl.float32)
    tl.store(out_ptr + indices, output)


def test_permute_dot_i1_subview_alignment():
    block = 32
    values = torch.arange(block * block, dtype=torch.float32).reshape(block, block)
    x_cpu = (values / 97.0).to(torch.float16)
    y_cpu = ((values.flip(0) + 11.0) / 89.0).to(torch.float16)
    expected = x_cpu.float() @ torch.minimum(y_cpu, torch.maximum(x_cpu, y_cpu).T).float()

    x = x_cpu.npu()
    y = y_cpu.npu()
    output = torch.empty((block, block), dtype=torch.float32, device="npu")
    permute_dot_alignment_kernel[(1,)](
        x,
        y,
        output,
        BLOCK=block,
        compile_mode="simd",
        multibuffer=False,
        num_stages=1,
        num_warps=4,
        use_bytecode=True,
        enable_mixed_cv=True,
        tile_mix_cube_loop=2,
        tile_mix_vector_loop=2,
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-3, atol=1e-3)
