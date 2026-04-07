# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for CCECodegen class."""

import pypto.language as pl
import pypto.language.op.manual as plm
import pytest
from pypto import DataType, backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir.builder import IRBuilder
from pypto.ir.op import block
from pypto.ir.pass_manager import PassManager


class TestCCECodegenBasics:
    """Test basic CCECodegen functionality."""

    def test_create_cce_codegen(self):
        """Test creating a CCECodegen instance."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)
        generator = codegen.CCECodegen()
        assert generator is not None

    def test_tadds_example(self):
        """Test generating code for a simple tensor addition with scalar example."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)
        ib = IRBuilder()

        with ib.function("test_tadds_simple") as f:
            # Define input and output parameters (Global Tensors -> DDR)
            input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
            input_b = f.param("input_b", ir.ScalarType(DataType.FP32))
            output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            # Constants for tile
            tile_height = 128
            tile_width = 128

            # Load (should infer input_a as DDR)
            tile_a = ib.let("tile_a", block.load(input_a, [0, 0], [tile_height, tile_width]))

            # Compute (UB)
            tile_sum = ib.let("tile_sum", block.adds(tile_a, input_b))

            # Store (should infer output as DDR)
            result = ib.let("result", block.store(tile_sum, [0, 0], [tile_height, tile_width], output))

            ib.return_stmt(result)

        func = f.get_result()
        program = ir.Program([func], "test_tadd_simple", ir.Span.unknown())

        pm = PassManager.get_strategy()
        optimized_program = pm.run_passes(program)

        generator = codegen.CCECodegen()
        files = generator.generate(optimized_program)
        kernel_name = list(optimized_program.functions.values())[0].name
        code = files["kernels/aiv/" + kernel_name + ".cpp"]

        # Verify function parameters unpacking and declarations are generated
        assert "GlobalTensor<float" in code
        assert "__gm__ Tensor*" in code
        assert "->buffer.addr" in code
        assert "union { uint64_t u64; float val; }" in code
        assert "float input_b_0 =" in code
        assert "GlobalType" in code  # Check for GlobalType suffix (e.g., output_0GlobalType)

        # Verify Tile type definitions are generated
        expected = "Tile<TileType::Vec, float, 128, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512>"
        assert expected in code
        assert "Type tile_" in code  # Check for tile type declarations with suffix
        assert "TASSIGN(tile_" in code

        # Verify instructions are generated
        assert "TLOAD(tile_" in code
        assert "TADDS(tile_" in code
        assert "TSTORE(" in code


def test_manual_fillpad_codegen_uses_destination_pad_value():
    """CCE manual.fillpad should bind a null-pad alias source before TFILLPAD."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualFillPadCCEProgram:
        @pl.function
        def fillpad_dynamic_tile(
            self,
            input: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Tensor[[16, 16], pl.FP32],
            rows_arg: pl.Scalar[pl.INDEX],
            cols_arg: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            src_type = plm.TileType(
                shape=[16, 16],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                pad=plm.TilePad.zero,
            )
            dst_type = plm.TileType(
                shape=[16, 16],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                pad=plm.TilePad.zero,
            )
            src = plm.make_tile(src_type, addr=0x0000, size=1024)
            dst = plm.make_tile(dst_type, addr=0x1000, size=1024)
            plm.set_validshape(src, rows_arg, cols_arg)
            plm.fillpad(dst, src)
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualFillPadCCEProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    kernel_name = list(optimized_program.functions.values())[0].name
    code = files["kernels/aiv/" + kernel_name + ".cpp"]

    assert "TFILLPAD(" in code
    assert "PadValue::Zero" in code
    assert "Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>" in code
    assert "using __manual_fillpad_src_alias_type_" in code
    assert ".GetValidRow(), src.GetValidCol()" in code
    assert "TASSIGN(__manual_fillpad_src_alias_" in code
    assert "TMOV(__manual_fillpad_src_alias_" not in code
    assert "TFILLPAD(dst, __manual_fillpad_src_alias_" in code


def test_manual_fillpad_inplace_codegen_uses_null_pad_temp_and_updates_valid_shape():
    """CCE fillpad(src, src) should lower to TFILLPAD_INPLACE with a null-pad alias source."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualFillPadInplaceCCEProgram:
        @pl.function
        def fillpad_same_tile(
            self,
            input: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Tensor[[16, 16], pl.FP32],
            rows_arg: pl.Scalar[pl.INDEX],
            cols_arg: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            tile_type = plm.TileType(
                shape=[16, 16],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                pad=plm.TilePad.zero,
            )
            src = plm.make_tile(tile_type, addr=0x0000, size=1024)
            plm.set_validshape(src, rows_arg, cols_arg)
            plm.fillpad(src, src)
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualFillPadInplaceCCEProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    kernel_name = list(optimized_program.functions.values())[0].name
    code = files["kernels/aiv/" + kernel_name + ".cpp"]

    assert "using __manual_fillpad_src_alias_type_" in code
    assert "TASSIGN(__manual_fillpad_src_alias_" in code
    assert "TMOV(__manual_fillpad_src_alias_" not in code
    assert "TFILLPAD_INPLACE(src, __manual_fillpad_src_alias_" in code
    assert "src.SetValidShape(16, 16);" in code


def test_manual_fillpad_expand_codegen_uses_destination_pad_value():
    """CCE manual.fillpad_expand should lower to TFILLPAD_EXPAND with dst TilePad encoded in the tile type."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class ManualFillPadExpandCCEProgram:
        @pl.function
        def fillpad_expand_dynamic_tile(
            self,
            input: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Tensor[[16, 32], pl.FP32],
            rows_arg: pl.Scalar[pl.INDEX],
            cols_arg: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 32], pl.FP32]:
            src_type = plm.TileType(
                shape=[16, 16],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                pad=plm.TilePad.zero,
            )
            dst_type = plm.TileType(
                shape=[16, 32],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                pad=plm.TilePad.zero,
            )
            src = plm.make_tile(src_type, addr=0x0000, size=1024)
            dst = plm.make_tile(dst_type, addr=0x1000, size=2048)
            plm.set_validshape(src, rows_arg, cols_arg)
            plm.fillpad_expand(dst, src)
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(ManualFillPadExpandCCEProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    kernel_name = list(optimized_program.functions.values())[0].name
    code = files["kernels/aiv/" + kernel_name + ".cpp"]

    assert "TFILLPAD_EXPAND(" in code
    assert "PadValue::Zero" in code
    assert "Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>" in code


class TestControlFlowCodegen:
    """Test control flow statement code generation."""

    def test_simple_for_loop(self):
        """Test simple for loop without iter_args."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)
        ib = IRBuilder()

        with ib.function("test_simple_for") as f:
            # Parameters
            input_tensor = f.param("input", ir.TensorType([128, 64], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 64], DataType.FP32))
            f.return_type(ir.TensorType([128, 64], DataType.FP32))

            # Loop variable
            i = ib.var("i", ir.ScalarType(DataType.INT32))

            # Simple for loop: for i in range(0, 4, 1)
            with ib.for_loop(i, 0, 4, 1):
                # Load tile inside loop
                tile_x = ib.let("tile_x", block.load(input_tensor, [i, 0], [32, 64]))
                # Store tile back
                result = ib.let("result", block.store(tile_x, [i, 0], [32, 64], output_tensor))

            ib.return_stmt(result)

        func = f.get_result()
        program = ir.Program([func], "test_simple_for", ir.Span.unknown())
        generator = codegen.CCECodegen()
        files = generator.generate(program)
        code = files["kernels/aiv/test_simple_for.cpp"]

        # Verify for loop structure
        assert "for (uint64_t i = 0; i < 4; i += 1) {" in code
        assert "TLOAD(tile_x, inputGlobal)" in code
        assert "TSTORE(outputGlobal, tile_x)" in code

    def test_nested_for_loops(self):
        """Test nested for loops."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)
        ib = IRBuilder()

        with ib.function("test_nested_for") as f:
            # Parameters
            input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
            output_tensor = f.param("output", ir.TensorType([128, 128], DataType.FP32))
            f.return_type(ir.TensorType([128, 128], DataType.FP32))

            # Outer loop variable
            i = ib.var("i", ir.ScalarType(DataType.INT32))
            # Inner loop variable
            j = ib.var("j", ir.ScalarType(DataType.INT32))

            # Nested for loops
            with ib.for_loop(i, 0, 4, 1):
                with ib.for_loop(j, 0, 4, 1):
                    # Load tile inside inner loop
                    tile_x = ib.let("tile_x", block.load(input_tensor, [i, j], [32, 32]))
                    # Store tile back
                    result = ib.let("result", block.store(tile_x, [i, j], [32, 32], output_tensor))

            ib.return_stmt(result)

        func = f.get_result()
        program = ir.Program([func], "test_nested_for", ir.Span.unknown())
        generator = codegen.CCECodegen()
        files = generator.generate(program)
        code = files["kernels/aiv/test_nested_for.cpp"]

        # Verify nested loop structure
        assert "for (uint64_t i = 0; i < 4; i += 1) {" in code
        assert "for (uint64_t j = 0; j < 4; j += 1) {" in code
        # Verify proper nesting (inner loop should appear after outer loop)
        assert code.index("for (uint64_t i") < code.index("for (uint64_t j")

    def test_if_statement_simple(self):
        """Test simple if statement code generation."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)
        span = ir.Span.unknown()

        # Build if statement directly using IR nodes
        condition = ir.ConstBool(True, span)

        # Then body: just an assignment
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        then_assign = ir.AssignStmt(x, ir.ConstInt(5, DataType.INT32, span), span)

        # Create if statement without else
        if_stmt = ir.IfStmt(condition, then_assign, None, [], span)

        # Create a simple function with the if statement
        ret_stmt = ir.ReturnStmt([], span)
        seq = ir.SeqStmts([if_stmt, ret_stmt], span)

        func = ir.Function("test_if", [], [ir.TensorType([1], DataType.FP32)], seq, span)
        program = ir.Program([func], "test_if", ir.Span.unknown())

        generator = codegen.CCECodegen()
        files = generator.generate(program)
        code = files["kernels/aiv/test_if.cpp"]

        # Verify if structure
        assert "if (true) {" in code or "if (1) {" in code
        assert "auto x = 5;" in code

    def test_if_else_statement(self):
        """Test if-else statement code generation."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)
        span = ir.Span.unknown()

        # Build condition
        a = ir.Var("a", ir.ScalarType(DataType.INT32), span)
        b = ir.Var("b", ir.ScalarType(DataType.INT32), span)
        condition = ir.Lt(a, b, DataType.INT32, span)

        # Then body
        x = ir.Var("x", ir.ScalarType(DataType.INT32), span)
        then_assign = ir.AssignStmt(x, ir.ConstInt(1, DataType.INT32, span), span)

        # Else body
        y = ir.Var("y", ir.ScalarType(DataType.INT32), span)
        else_assign = ir.AssignStmt(y, ir.ConstInt(2, DataType.INT32, span), span)

        # Create if-else statement
        if_stmt = ir.IfStmt(condition, then_assign, else_assign, [], span)

        # Create function
        # First assign a and b
        assign_a = ir.AssignStmt(a, ir.ConstInt(5, DataType.INT32, span), span)
        assign_b = ir.AssignStmt(b, ir.ConstInt(10, DataType.INT32, span), span)
        ret_stmt = ir.ReturnStmt([], span)
        seq = ir.SeqStmts([assign_a, assign_b, if_stmt, ret_stmt], span)

        func = ir.Function("test_if_else", [], [ir.TensorType([1], DataType.FP32)], seq, span)
        program = ir.Program([func], "test_if_else", ir.Span.unknown())

        generator = codegen.CCECodegen()
        files = generator.generate(program)
        code = files["kernels/aiv/test_if_else.cpp"]

        # Verify if-else structure
        assert "if ((a < b)) {" in code or "if (a < b) {" in code
        assert "} else {" in code
        assert "auto x = 1;" in code
        assert "auto y = 2;" in code


class TestMatmulCodegen:
    """Test matrix multiplication code generation."""

    def test_matmul_simple(self):
        """Test simple matmul with correct TileTypes for different memory spaces."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class TestMatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def test_matmul(
                self,
                a: pl.Tensor[[64, 64], pl.FP16],
                b: pl.Tensor[[64, 64], pl.FP16],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                """Test matmul with L1/Left/Right/Acc memory spaces."""
                # Load to L1 (Mat tiles), move to Left/Right, matmul
                tile_a_l1: pl.Tile[[64, 64], pl.FP16] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat
                )  # L1
                tile_b_l1: pl.Tile[[64, 64], pl.FP16] = pl.load(
                    b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat
                )

                # Move to compute memory (Left, Right)
                tile_a_l0a: pl.Tile[[64, 64], pl.FP16] = pl.move(
                    tile_a_l1, target_memory=pl.MemorySpace.Left
                )  # Left
                tile_b_l0b: pl.Tile[[64, 64], pl.FP16] = pl.move(
                    tile_b_l1, target_memory=pl.MemorySpace.Right
                )  # Right

                # Matmul
                tile_c_l0c: pl.Tile[[64, 64], pl.FP32] = pl.matmul(tile_a_l0a, tile_b_l0b)

                # Move back and store
                # don't use TMOV to move l0c to l1, it has some constraints on the tile type(to be fixed)
                # TSTORE can support l0c to GM
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c_l0c, [0, 0], [64, 64], c)
                return result

        program = TestMatmulProgram

        pm = PassManager.get_strategy()
        optimized_program = pm.run_passes(program)

        generator = codegen.CCECodegen()
        files = generator.generate(optimized_program)
        code = files["kernels/aic/test_matmul.cpp"]

        # Verify TileTypes based on memory space
        assert "Tile<TileType::Mat" in code  # For L1 tiles
        assert "Tile<TileType::Left" in code  # For Left tile
        assert "Tile<TileType::Right" in code  # For Right tile
        assert "Tile<TileType::Acc" in code  # For Acc tile

        # Verify instructions
        assert "TMOV(" in code
        assert "TMATMUL(" in code

    def test_matmul_acc(self):
        """Test accumulating matmul operation."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)

        @pl.program
        class TestMatmulAccProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def test_matmul_acc(
                self,
                a0: pl.Tensor[[32, 32], pl.FP16],
                a1: pl.Tensor[[32, 32], pl.FP16],
                b0: pl.Tensor[[32, 32], pl.FP16],
                b1: pl.Tensor[[32, 32], pl.FP16],
                c: pl.Tensor[[32, 32], pl.FP32],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                """Test accumulating matmul operation."""
                # Load tiles to L1 and move to compute buffers
                tile_a0_l1: pl.Tile[[32, 32], pl.FP16] = pl.load(
                    a0, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
                )
                tile_b0_l1: pl.Tile[[32, 32], pl.FP16] = pl.load(
                    b0, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
                )
                tile_a0_l0a: pl.Tile[[32, 32], pl.FP16] = pl.move(
                    tile_a0_l1, target_memory=pl.MemorySpace.Left
                )
                tile_b0_l0b: pl.Tile[[32, 32], pl.FP16] = pl.move(
                    tile_b0_l1, target_memory=pl.MemorySpace.Right
                )

                # First matmul
                tile_c0: pl.Tile[[32, 32], pl.FP32] = pl.matmul(tile_a0_l0a, tile_b0_l0b)

                # Load second batch
                tile_a1_l1: pl.Tile[[32, 32], pl.FP16] = pl.load(
                    a1, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
                )
                tile_b1_l1: pl.Tile[[32, 32], pl.FP16] = pl.load(
                    b1, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
                )
                tile_a1_l0a: pl.Tile[[32, 32], pl.FP16] = pl.move(
                    tile_a1_l1, target_memory=pl.MemorySpace.Left
                )
                tile_b1_l0b: pl.Tile[[32, 32], pl.FP16] = pl.move(
                    tile_b1_l1, target_memory=pl.MemorySpace.Right
                )

                # Accumulating matmul
                tile_c1: pl.Tile[[32, 32], pl.FP32] = pl.matmul_acc(tile_c0, tile_a1_l0a, tile_b1_l0b)

                # Move result and store
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_c1, [0, 0], [32, 32], c)
                return result

        program = TestMatmulAccProgram

        pm = PassManager.get_strategy()
        optimized_program = pm.run_passes(program)

        generator = codegen.CCECodegen()
        files = generator.generate(optimized_program)
        code = files["kernels/aic/test_matmul_acc.cpp"]

        # Verify both TMATMUL and TMATMUL_ACC are generated
        assert "TMATMUL(" in code
        assert "TMATMUL_ACC(" in code


def test_debug_dump_tensor_dynamic_shape_codegen():
    """CCE debug.dump_tensor should emit a runtime GlobalTensor view for dynamic shapes."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    M = pl.DynVar("M")

    @pl.program
    class DebugDumpTensorShapeProgram:
        @pl.function
        def debug_dump_tensor_shape(
            self,
            input: pl.Tensor[[M, 32], pl.FP32],
            output: pl.Tensor[[M, 32], pl.FP32],
        ):
            rows = pl.tensor.dim(input, 0)
            plm.dump_tensor(input, offsets=[0, 0], shapes=[rows, 16])
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(DebugDumpTensorShapeProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    code = files["kernels/aiv/debug_dump_tensor_shape.cpp"]

    assert "using __debug_dump_tensor_shape_" in code
    assert "Shape<1, 1, 1, -1, 16>" in code
    assert "GlobalTensor<float" in code
    assert "TPRINT(__debug_dump_tensor_view_" in code


def test_debug_dump_tensor_dynamic_window_codegen():
    """CCE debug.dump_tensor should preserve dynamic offsets in the runtime view."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)
    M = pl.DynVar("M")

    @pl.program
    class DebugDumpTensorWindowProgram:
        @pl.function
        def debug_dump_tensor_window(
            self,
            input: pl.Tensor[[M, 32], pl.FP32],
            row_off: pl.Scalar[pl.INDEX],
            output: pl.Tensor[[M, 32], pl.FP32],
        ):
            rows = pl.tensor.dim(input, 0)
            plm.dump_tensor(input, offsets=[row_off, 0], shapes=[rows, 16])
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(DebugDumpTensorWindowProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    code = files["kernels/aiv/debug_dump_tensor_window.cpp"]

    assert "__debug_dump_tensor_view_" in code
    assert "TPRINT(__debug_dump_tensor_view_" in code
    assert " + (" in code or " + " in code


def test_debug_dump_tile_dynamic_offset_codegen():
    """CCE dump_tile window lowering should emit runtime clamp logic and direct printing."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.CCE)

    @pl.program
    class DebugDumpTileOffsetProgram:
        @pl.function
        def debug_dump_tile_offset(
            self,
            input: pl.Tensor[[32, 32], pl.FP32],
            row_off: pl.Scalar[pl.INDEX],
            output: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            plm.dump_tile(tile, offsets=[row_off, 0], shapes=[8, 16])
            return output

    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(DebugDumpTileOffsetProgram)

    generator = codegen.CCECodegen()
    files = generator.generate(optimized_program)
    code = files["kernels/aiv/debug_dump_tile_offset.cpp"]

    assert "GetValidRow()" in code
    assert "GetValidCol()" in code
    assert "pto::GetTileOffset" in code
    assert "pto::PrintValue(" in code
    assert 'cce::printf("=== [TPRINT Tile Window]' in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
