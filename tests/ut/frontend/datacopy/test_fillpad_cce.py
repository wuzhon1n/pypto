# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime datacopy example for plm.fillpad on CCE.

CCE manual.load lowers to a bare ``TLOAD(tile, tensor)``. For ND row-major vec
tiles, that means we must first load the full physical tile and only then
narrow the runtime valid-shape before ``fillpad``.
"""

import torch
import torch_npu

import pypto.frontend as fe
import pypto.language as pl
import pypto.language.op.manual as plm


@fe.kernel
def fillpad_dynamic_cce_kernel(
    x: pl.Tensor[[8, 8], pl.INT32],
    z: pl.Tensor[[16, 8], pl.INT32],
) -> pl.Tensor[[16, 8], pl.INT32]:
    src_type = plm.TileType(
        shape=[8, 8],
        dtype=pl.INT32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
        pad=plm.TilePad.zero,
    )
    dst_type = plm.TileType(
        shape=[8, 8],
        dtype=pl.INT32,
        target_memory=pl.MemorySpace.Vec,
        pad=plm.TilePad.zero,
    )
    src = plm.make_tile(src_type, addr=0x0000, size=256)
    dst = plm.make_tile(dst_type, addr=0x0100, size=256)

    with pl.section_vector():
        plm.load(src, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        plm.set_validshape(src, 5, 7)
        plm.dump_tile(src)

        plm.fillpad(dst, src)
        plm.dump_tile(dst)
        plm.fillpad(src, src)
        plm.dump_tile(src)

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(z, dst, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        plm.store(z, src, [8, 0])
        pl.system.bar_all()

    return z


@fe.jit()
def test_fillpad_dynamic_cce():
    compiled_lib = fe.compile(fillpad_dynamic_cce_kernel, arch="a3", codegen_mode="cce")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:0"
    torch.npu.set_device(device)

    x = torch.full((8, 8), -99, device=device, dtype=torch.int32)
    x[:5, :7] = torch.arange(35, device=device, dtype=torch.int32).reshape(5, 7)
    z = torch.empty((16, 8), device=device, dtype=torch.int32)

    fe.launch(None, 1, compiled_lib, x, z)
    torch.npu.synchronize()

    print("***********cce padded output***********")
    print(z.shape, z.dtype)
    print(z)

    z_ref = torch.zeros((8, 8), device=device, dtype=torch.int32)
    z_ref[:5, :7] = x[:5, :7]
    print("***********golden padded output***********")
    print(z_ref.shape, z_ref.dtype)
    print(z_ref)

    torch.testing.assert_close(z[:8, :], z_ref)
    torch.testing.assert_close(z[8:, :], z_ref)
    print("result equal!")


if __name__ == "__main__":
    test_fillpad_dynamic_cce()
    print("\nAll tests passed!")
