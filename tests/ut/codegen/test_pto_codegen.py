# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for PTOCodegen - MLIR generation from PyPTO IR.

The new PTOCodegen generates PTO-ISA MLIR dialect instead of PTO assembly.
Tests verify:
- Correct MLIR module structure
- Proper function signatures with tensor pointers
- make_tensor_view generation for tensor parameters
- alloc_tile generation for tile buffers
- Operator lowering (block.load/store/mul/adds -> pto.tload/tstore/tmul/tadds)
- SSA form with correct variable naming
"""

from dataclasses import dataclass
import pypto.language as pl
import pypto.language.op.manual as plm
import pytest
from pypto import DataType, backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir import OptimizationStrategy, PassManager
from pypto.ir.builder import IRBuilder
from pypto.ir.op import block, debug as debug_op
from pypto.ir.pto_codegen import (
    _generate_arg_unpacking,
    _generate_kernel_wrapper,
    _preprocess_ptoas_output,
    generate,
)

from pypto.language.typing.tiling import Array

PTOCodegen = codegen.PTOCodegen

# Dynamic shape variables for wrapper dispatch tests
# pyright: reportUndefinedVariable=false
_TH = pl.dynamic("TH")
_TW = pl.dynamic("TW")


@pl.program
class _DynKernel:
    """Dynamic shape kernel used in wrapper dispatch tests."""

    @pl.function(type=pl.FunctionType.InCore)
    def dyn_func(
        self,
        a: pl.Tensor[[_TH, _TW], pl.FP32],
        b: pl.Tensor[[_TH, _TW], pl.FP32],
        output: pl.Tensor[[_TH, _TW], pl.FP32],
    ) -> pl.Tensor[[_TH, _TW], pl.FP32]:
        a_tile = pl.load(a, [0, 0], [128, 128])
        b_tile = pl.load(b, [0, 0], [128, 128])
        result = pl.add(a_tile, b_tile)
        return pl.store(result, [0, 0], [128, 128], output)


def _get_dyn_incore_func():
    """Return the transformed InCore function from _DynKernel."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)
    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed = pm.run_passes(_DynKernel)
    for func in transformed.functions.values():
        if func.func_type == ir.FunctionType.InCore:
            return func
    raise RuntimeError("No InCore function found in _DynKernel")


def _get_mlir_code(result):
    """Normalize generate() result to MLIR string (support both str and dict)."""
    return result if isinstance(result, str) else "".join(result.values())


SAMPLE_PTOAS_OUTPUT = """\
#include "pto/pto-inst.hpp"
using namespace pto;

\ttemplate <typename To, typename From>
\tstatic inline To ptoas_bitcast(From from) {
\t  static_assert(sizeof(To) == sizeof(From), "ptoas_bitcast: size mismatch");
\t  To to;
\t  __builtin_memcpy(&to, &from, sizeof(To));
\t  return to;
\t}
\t
__global__ AICORE void test_func(__gm__ float* v1, float v2, __gm__ float* v3) {
  TLOAD(v1);
  TADDS(v2);
  TSTORE(v3);
  return;
}
"""


def _make_func(name, params_spec):
    """Build a Function from parameter specs.

    Args:
        name: Function name.
        params_spec: list of (param_name, "tensor"|"scalar") tuples.

    Returns:
        ir.Function with InCore type.
    """
    ib = IRBuilder()
    with ib.function(name, type=ir.FunctionType.InCore) as f:
        param_vars = []
        for pname, kind in params_spec:
            if kind == "tensor":
                param_vars.append(f.param(pname, ir.TensorType([16, 16], DataType.FP32)))
            elif kind == "scalar":
                param_vars.append(f.param(pname, ir.ScalarType(DataType.FP32)))

        # Minimal body: load first tensor param → store
        tensor_params = [v for v, (_, k) in zip(param_vars, params_spec) if k == "tensor"]
        if len(tensor_params) >= 2:
            t = ib.let("t", block.load(tensor_params[0], [0, 0], [16, 16]))
            result = ib.let("result", block.store(t, [0, 0], [16, 16], tensor_params[-1]))
            f.return_type(ir.TensorType([16, 16], DataType.FP32))
            ib.return_stmt(result)
        elif len(tensor_params) == 1:
            t = ib.let("t", block.load(tensor_params[0], [0, 0], [16, 16]))
            result = ib.let("result", block.store(t, [0, 0], [16, 16], tensor_params[0]))
            f.return_type(ir.TensorType([16, 16], DataType.FP32))
            ib.return_stmt(result)
        else:
            f.return_type(ir.ScalarType(DataType.FP32))
            ib.return_stmt(param_vars[0])

    return f.get_result()


def test_pto_codegen_basic_mlir_structure():
    """Test that PTOCodegen generates valid MLIR module structure."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class BasicProgram:
        @pl.function
        def test_func(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.add(tile_a, 1.0)
            pl.store(tile_b, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

    # Compile with PTOAS strategy (applies necessary passes + codegen)
    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(BasicProgram)

    # Generate MLIR
    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify MLIR module structure
    assert "module {" in mlir_code
    assert "func.func @test_func" in mlir_code
    assert "return" in mlir_code
    assert "}" in mlir_code

def test_pto_codegen_tiling():
    """Test that PTOCodegen generates correct Tiling."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @dataclass
    class Tiling:
        n: int
        m: int
        arr: Array[float, 3]
    @pl.program
    class TilingProgram:
        @pl.function
        def test_func(self,
            x: pl.Tensor[[64, 64], pl.FP32],
            y: pl.Tensor[[64, 64], pl.FP32],
            tiling: Tiling,
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            n = tiling.n
            m = tiling.m
            tmp0 = tiling.arr[0]
            tmp1 = tiling.arr[1]
            # current ptoas CodeGen not support dynamic offsets and shapes
            result0 = n + m
            result1 = tmp0 * tmp1
            return x

    # Compile with PTOAS strategy (applies necessary passes + codegen)
    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(TilingProgram)

    # Generate MLIR
    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))
    assert("%arg2: index, %arg3: index, %arg4: f32, %arg5: f32, %arg6: f32" in mlir_code)
    assert("arith.addi %arg2, %arg3 : index" in mlir_code)
    assert("arith.mulf %arg4, %arg5 : f32" in mlir_code)


def test_pto_codegen_tensor_parameters():
    """Test that tensor parameters generate correct make_tensor_view."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class TensorParamProgram:
        @pl.function
        def tensor_param_func(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ):
            tile_a = pl.load(input_a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.load(input_b, offsets=[0, 0], shapes=[32, 32])
            tile_c = pl.mul(tile_a, tile_b)
            pl.store(tile_c, offsets=[0, 0], shapes=[32, 32], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(TensorParamProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify function signature with pointer types
    assert "%arg0: !pto.ptr<f32>" in mlir_code
    assert "%arg1: !pto.ptr<f32>" in mlir_code
    assert "%arg2: !pto.ptr<f32>" in mlir_code

    # Verify make_tensor_view generation
    assert "pto.make_tensor_view" in mlir_code
    assert "shape = [%c64, %c64]" in mlir_code or "shape = [%c32, %c32]" in mlir_code
    assert "strides = " in mlir_code
    assert "!pto.tensor_view<?x?xf32>" in mlir_code


def test_debug_dump_tensor_ir_returns_unknown_type():
    """debug.dump_tensor IR helper should return UnknownType."""
    span = ir.Span.unknown()
    tensor_var = ir.Var("input", ir.TensorType([48, 64], DataType.FP32), span)

    call = debug_op.dump_tensor(tensor_var)

    assert isinstance(call.type, ir.UnknownType)


def test_debug_dump_tile_ir_returns_unknown_type():
    """debug.dump_tile IR helper should return UnknownType."""
    span = ir.Span.unknown()
    tile_var = ir.Var("tile", ir.TileType([16, 16], DataType.FP32), span)

    call = debug_op.dump_tile(tile_var)

    assert isinstance(call.type, ir.UnknownType)


def test_debug_printf_ir_returns_unknown_type():
    """debug.printf IR helper should return UnknownType."""
    span = ir.Span.unknown()
    scalar_var = ir.Var("value", ir.ScalarType(DataType.INT32), span)

    call = debug_op.printf("value=%d", scalar_var)

    assert isinstance(call.type, ir.UnknownType)


def test_debug_assert_ir_returns_unknown_type():
    """debug.assert IR helper should return UnknownType."""
    span = ir.Span.unknown()
    flag_var = ir.Var("flag", ir.ScalarType(DataType.BOOL), span)
    value_var = ir.Var("value", ir.ScalarType(DataType.INT32), span)

    call = debug_op.assert_(flag_var)
    assert isinstance(call.type, ir.UnknownType)

    call = debug_op.assert_(flag_var, "value=%d", value_var, condition_text="flag")
    assert isinstance(call.type, ir.UnknownType)


def test_debug_trap_ir_returns_unknown_type():
    """debug.trap IR helper should return UnknownType."""
    call = debug_op.trap()

    assert isinstance(call.type, ir.UnknownType)


def test_debug_printf_extended_types_return_unknown_type():
    """debug.printf should accept signed/unsigned/index/bool types per v1 policy."""
    span = ir.Span.unknown()
    int_var = ir.Var("int_value", ir.ScalarType(DataType.INT32), span)
    bool_var = ir.Var("bool_value", ir.ScalarType(DataType.BOOL), span)
    index_var = ir.Var("index_value", ir.ScalarType(DataType.INDEX), span)
    uint_var = ir.Var("uint_value", ir.ScalarType(DataType.UINT32), span)
    fp32_var = ir.Var("fp32_value", ir.ScalarType(DataType.FP32), span)

    call = debug_op.printf("%i %d %x %u %f", int_var, bool_var, index_var, uint_var, fp32_var)

    assert isinstance(call.type, ir.UnknownType)


def test_debug_printf_python_bool_returns_unknown_type():
    """debug.printf should accept direct Python bool for %d/%u/%i."""
    call = debug_op.printf("flag=%d", True)

    assert isinstance(call.type, ir.UnknownType)

    call_unsigned = debug_op.printf("flag=%u", True)

    assert isinstance(call_unsigned.type, ir.UnknownType)


def test_debug_printf_pure_text_returns_unknown_type():
    """debug.printf should accept pure text with zero scalar arguments."""
    call = debug_op.printf("hello world")

    assert isinstance(call.type, ir.UnknownType)


def test_pto_codegen_alloc_tile():
    """Test that tile buffers generate alloc_tile operations."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class AllocTileProgram:
        @pl.function
        def alloc_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_c = pl.mul(tile_a, tile_b)
            pl.store(tile_c, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(AllocTileProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify alloc_tile operations
    assert "pto.alloc_tile" in mlir_code
    assert "loc=vec" in mlir_code  # Vector buffer (PTO address space)
    assert "dtype=f32" in mlir_code
    assert "rows=32, cols=32" in mlir_code


def test_pto_codegen_printf_lowering():
    """plm.printf should lower to one pto.print per scalar argument."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class PrintfProgram:
        @pl.function
        def printf_test(
            self,
            input: pl.Tensor[[16, 16], pl.INT32],
            output: pl.Tensor[[16, 16], pl.INT32],
            x: pl.Scalar[pl.INT32],
            y: pl.Scalar[pl.UINT32],
            z: pl.Scalar[pl.FP32],
        ):
            plm.printf("x=%d y=%x z=%f\n", x, y, z)
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(PrintfProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))
    assert mlir_code.count("pto.print") == 3
    assert 'pto.print ins("x=%d", ' in mlir_code
    assert 'pto.print ins(" y=%llx", ' in mlir_code
    assert 'pto.print ins(" z=%f\\n", ' in mlir_code


def test_pto_codegen_printf_pure_text_lowering():
    """plm.printf should lower pure text to a single pto.print with a dummy scalar."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class PrintfTextProgram:
        @pl.function
        def printf_text_test(
            self,
            input: pl.Tensor[[16, 16], pl.INT32],
            output: pl.Tensor[[16, 16], pl.INT32],
        ):
            plm.printf("hello world\n")
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(PrintfTextProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert mlir_code.count("pto.print") == 1
    assert 'pto.print ins("hello world\\n", ' in mlir_code
    assert "arith.constant 0 : i32" in mlir_code


def test_pto_codegen_trap_lowering():
    """plm.trap should lower directly to pto.trap."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class TrapProgram:
        @pl.function
        def trap_test(
            self,
            input: pl.Tensor[[16, 16], pl.INT32],
            output: pl.Tensor[[16, 16], pl.INT32],
        ):
            plm.trap()
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(TrapProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert mlir_code.count("pto.trap") == 1
    assert "pto.print" not in mlir_code
    assert "pto.tprint" not in mlir_code


def test_pto_codegen_assert_lowering():
    """plm.assert_ should lower to scf.if with prefixed pto.print and pto.trap in the failure branch."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class AssertProgram:
        @pl.function
        def assert_test(
            self,
            input: pl.Tensor[[16, 16], pl.INT32],
            output: pl.Tensor[[16, 16], pl.INT32],
            flag: pl.Scalar[pl.BOOL],
            x: pl.Scalar[pl.INT32],
        ):
            plm.assert_(flag)
            plm.assert_(x > 0, "x=%d", x)
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(AssertProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert mlir_code.count("arith.cmpi eq") >= 2
    assert mlir_code.count("scf.if") == 2
    assert "pto.print ins(\"[ASSERT] Assertion 'flag'\\n\", " in mlir_code
    assert "pto.print ins(\"[ASSERT] Assertion 'x > 0', x=%d\\n\", " in mlir_code
    assert mlir_code.count("pto.trap") == 2


def test_pto_codegen_printf_accepts_common_flags_width_precision():
    """plm.printf should accept common flags, width, and precision syntax."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class PrintfModifierProgram:
        @pl.function
        def printf_modifier_test(
            self,
            input: pl.Tensor[[16, 16], pl.INT32],
            output: pl.Tensor[[16, 16], pl.INT32],
            hex_value: pl.Scalar[pl.UINT32],
            float_value: pl.Scalar[pl.FP32],
        ):
            plm.printf("hex=%#08x f=%+08.3f\n", hex_value, float_value)
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(PrintfModifierProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))
    print(mlir_code)
    assert 'pto.print ins("hex=%#08llx", ' in mlir_code
    assert 'pto.print ins(" f=%+08.3f\\n", ' in mlir_code


def test_pto_codegen_preserves_unsigned_mlir_types_outside_printf():
    """Non-printf PTO codegen should keep unsigned MLIR types instead of forcing signless."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class UnsignedTypeProgram:
        @pl.function
        def unsigned_type_test(
            self,
            input: pl.Tensor[[16, 16], pl.UINT8],
            output: pl.Tensor[[16, 16], pl.UINT8],
            scalar: pl.Scalar[pl.UINT32],
        ) -> pl.Scalar[pl.UINT32]:
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=output)
            return scalar

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(UnsignedTypeProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert "!pto.ptr<ui8>" in mlir_code
    assert "ui32" in mlir_code


def test_pto_codegen_block_load_lowering():
    """Test that block.load generates partition_view + tload."""

    @pl.program
    class LoadProgram:
        @pl.function
        def load_test(self, input: pl.Tensor[[64, 64], pl.FP32], output: pl.Tensor[[64, 64], pl.FP32]):
            tile = pl.load(input, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(LoadProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify partition_view generation
    assert "pto.partition_view" in mlir_code
    assert "offsets = [%c0, %c0]" in mlir_code
    assert "sizes = [%c32, %c32]" in mlir_code
    assert "!pto.partition_tensor_view<32x32xf32>" in mlir_code

    # Verify tload generation
    assert "pto.tload" in mlir_code
    assert "ins(" in mlir_code
    assert "outs(" in mlir_code
    assert "!pto.tile_buf<" in mlir_code


def test_pto_codegen_block_store_lowering():
    """Test that block.store generates partition_view + tstore."""

    @pl.program
    class StoreProgram:
        @pl.function
        def store_test(self, input: pl.Tensor[[32, 32], pl.FP32], output: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(input, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(StoreProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify tstore generation
    assert "pto.tstore" in mlir_code
    assert "ins(" in mlir_code
    assert "outs(" in mlir_code


def test_pto_codegen_dump_tensor_full_tensor_lowering():
    """plm.dump_tensor without offsets/shapes prints the full logical tensor."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class DumpFullProgram:
        @pl.function
        def dump_full(
            self,
            input: pl.Tensor[[48, 64], pl.FP32],
            output: pl.Tensor[[48, 64], pl.FP32],
        ):
            plm.dump_tensor(input)
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(DumpFullProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert mlir_code.count("pto.tprint") == 1
    assert "sizes = [%c48, %c64]" in mlir_code
    assert "!pto.partition_tensor_view<48x64xf32>" in mlir_code


def test_pto_codegen_dump_tensor_static_window_lowering():
    """plm.dump_tensor with offsets/shapes lowers to partition_view + tprint for that window."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class DumpWindowProgram:
        @pl.function
        def dump_window(
            self,
            input: pl.Tensor[[48, 64], pl.FP32],
            output: pl.Tensor[[48, 64], pl.FP32],
        ):
            plm.dump_tensor(input, offsets=[8, 12], shapes=[20, 24])
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(DumpWindowProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert mlir_code.count("pto.tprint") == 1
    assert "offsets = [%c8, %c12]" in mlir_code
    assert "sizes = [%c20, %c24]" in mlir_code
    assert "!pto.partition_tensor_view<20x24xf32>" in mlir_code


def test_pto_codegen_dump_tile_lowering():
    """plm.dump_tile lowers directly to pto.tprint on the tile value."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class DumpTileProgram:
        @pl.function
        def dump_tile_test(
            self,
            input: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            plm.dump_tile(tile)
            pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(DumpTileProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert mlir_code.count("pto.tprint") == 1
    assert "pto.tprint ins(" in mlir_code
    assert "pto.tprint ins()" not in mlir_code


def test_pto_codegen_dump_tile_static_window_lowering():
    """plm.dump_tile with offsets/shapes lowers to pto.subset + pto.tprint."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class DumpTileWindowProgram:
        @pl.function
        def dump_tile_window_test(
            self,
            input: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            plm.dump_tile(tile, offsets=[4, 0], shapes=[8, 16])
            pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(DumpTileWindowProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert mlir_code.count("pto.tprint") == 1
    assert "pto.subset" in mlir_code
    assert "[%c4, %c0] sizes [8, 16]" in mlir_code


def test_pto_codegen_dump_tensor_dynamic_shape_lowering():
    """dump_tensor should lower dynamic shapes to partition views with dynamic result dims."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)
    M = pl.DynVar("M")

    @pl.program
    class DumpDynamicShapeProgram:
        @pl.function
        def dump_dynamic_shape(
            self,
            input: pl.Tensor[[M, 32], pl.FP32],
            output: pl.Tensor[[M, 32], pl.FP32],
        ):
            rows = pl.tensor.dim(input, 0)
            plm.dump_tensor(input, offsets=[0, 0], shapes=[rows, 16])
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(DumpDynamicShapeProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert "pto.partition_view" in mlir_code
    assert "!pto.partition_tensor_view<?x16xf32>" in mlir_code
    assert "sizes = [" in mlir_code


def test_pto_codegen_dump_tensor_dynamic_window_lowering():
    """dump_tensor should preserve dynamic offsets in partition_view lowering."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)
    M = pl.DynVar("M")

    @pl.program
    class DumpDynamicWindowProgram:
        @pl.function
        def dump_dynamic_window(
            self,
            input: pl.Tensor[[M, 32], pl.FP32],
            row_off: pl.Scalar[pl.INDEX],
            output: pl.Tensor[[M, 32], pl.FP32],
        ):
            rows = pl.tensor.dim(input, 0)
            plm.dump_tensor(input, offsets=[row_off, 0], shapes=[rows, 16])
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(DumpDynamicWindowProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert "pto.partition_view" in mlir_code
    assert "!pto.partition_tensor_view<?x16xf32>" in mlir_code
    assert "offsets = [" in mlir_code


def test_pto_codegen_dump_tile_dynamic_valid_shape_lowering():
    """dump_tile full dumps should preserve dynamic valid-shape tile types."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class DumpDynamicTileProgram:
        @pl.function
        def dump_dynamic_tile(
            self,
            rows: pl.Scalar[pl.INDEX],
            cols: pl.Scalar[pl.INDEX],
            output: pl.Tensor[[16, 16], pl.FP32],
        ):
            tile_type = plm.TileType(
                shape=[16, 16],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
            )
            tile = plm.make_tile(tile_type, addr=0x0000, size=1024)
            plm.set_validshape(tile, rows, cols)
            plm.dump_tile(tile)
            return output

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(DumpDynamicTileProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert "pto.set_validshape" in mlir_code
    assert "pto.tprint ins(" in mlir_code
    assert "v_row=?, v_col=?" in mlir_code


def test_pto_codegen_manual_fillpad_updates_pad_and_valid_shape():
    """plm.fillpad should lower through a null-pad alias view without changing the original src type."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class ManualFillPadProgram:
        @pl.function
        def fillpad_dynamic_tile(
            self,
            rows: pl.Scalar[pl.INDEX],
            cols: pl.Scalar[pl.INDEX],
            output: pl.Tensor[[16, 16], pl.FP32],
        ):
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
            plm.set_validshape(src, rows, cols)
            plm.fillpad(dst, src)
            plm.dump_tile(dst)
            return output

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(ManualFillPadProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert mlir_code.count("pto.alloc_tile") == 3
    assert mlir_code.count("pto.set_validshape") == 1
    assert "pto.alloc_tile addr = %addr0 valid_row = " in mlir_code
    assert " valid_col = " in mlir_code
    assert "pto.tmov ins(" not in mlir_code
    assert "pto.tfillpad ins(" in mlir_code
    assert "pad=0" in mlir_code
    assert "pad=1" in mlir_code
    assert "v_row=?, v_col=?" in mlir_code


def test_pto_codegen_manual_fillpad_expand_uses_expand_op():
    """plm.fillpad_expand should lower through a null-pad alias view."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class ManualFillPadExpandProgram:
        @pl.function
        def fillpad_expand_dynamic_tile(
            self,
            rows: pl.Scalar[pl.INDEX],
            cols: pl.Scalar[pl.INDEX],
            output: pl.Tensor[[16, 32], pl.FP32],
        ):
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
            plm.set_validshape(src, rows, cols)
            plm.fillpad_expand(dst, src)
            plm.dump_tile(dst)
            return output

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(ManualFillPadExpandProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert mlir_code.count("pto.alloc_tile") == 3
    assert mlir_code.count("pto.set_validshape") == 1
    assert "pto.alloc_tile addr = %addr0 valid_row = " in mlir_code
    assert " valid_col = " in mlir_code
    assert "pto.tmov ins(" not in mlir_code
    assert "pto.tfillpad_expand ins(" in mlir_code
    assert "pad=0" in mlir_code
    assert "pad=1" in mlir_code
    assert "rows=16, cols=32" in mlir_code
    assert "v_row=16, v_col=32" in mlir_code


def test_pto_codegen_manual_fillpad_rejects_same_tile_inplace():
    """PTO backend should reject fillpad(src, src) until same-tile lowering is supported."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class ManualFillPadInplaceProgram:
        @pl.function
        def fillpad_same_tile(
            self,
            rows: pl.Scalar[pl.INDEX],
            cols: pl.Scalar[pl.INDEX],
            output: pl.Tensor[[16, 16], pl.FP32],
        ):
            tile_type = plm.TileType(
                shape=[16, 16],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                pad=plm.TilePad.zero,
            )
            src = plm.make_tile(tile_type, addr=0x0000, size=1024)
            plm.set_validshape(src, rows, cols)
            plm.fillpad(src, src)
            return output

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(ManualFillPadInplaceProgram)

    codegen_obj = PTOCodegen()
    with pytest.raises(ValueError, match="fillpad\\(src, src\\)"):
        codegen_obj.generate(transformed_program)


def test_manual_fillpad_rejects_invalid_pad_modes():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    with pytest.raises(TypeError):
        @pl.program
        class InvalidFillPadTypeProgram:
            @pl.function
            def invalid_fillpad_type(
                self,
                output: pl.Tensor[[16, 16], pl.FP32],
            ):
                tile_type = plm.TileType(
                    shape=[16, 16],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                    pad="zero",
                )
                src = plm.make_tile(tile_type, addr=0x0000, size=1024)
                dst = plm.make_tile(tile_type, addr=0x1000, size=1024)
                plm.fillpad(dst, src)
                return output

    with pytest.raises(ValueError):
        @pl.program
        class InvalidFillPadNullProgram:
            @pl.function
            def invalid_fillpad_null(
                self,
                output: pl.Tensor[[16, 16], pl.FP32],
            ):
                src_type = plm.TileType(
                    shape=[16, 16],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                )
                dst_type = plm.TileType(
                    shape=[16, 16],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                    pad=plm.TilePad.null,
                )
                src = plm.make_tile(src_type, addr=0x0000, size=1024)
                dst = plm.make_tile(dst_type, addr=0x1000, size=1024)
                plm.fillpad(dst, src)
                return output


def test_manual_fillpad_expand_rejects_invalid_shape_and_pad():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    with pytest.raises(ValueError, match="manual.fillpad_expand: out.tile_view.pad must not be TilePad.null"):
        @pl.program
        class InvalidFillPadExpandNullProgram:
            @pl.function
            def invalid_fillpad_expand_null(
                self,
                output: pl.Tensor[[16, 32], pl.FP32],
            ):
                src_type = plm.TileType(
                    shape=[16, 16],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                )
                dst_type = plm.TileType(
                    shape=[16, 32],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                    pad=plm.TilePad.null,
                )
                src = plm.make_tile(src_type, addr=0x0000, size=1024)
                dst = plm.make_tile(dst_type, addr=0x1000, size=2048)
                plm.fillpad_expand(dst, src)
                return output

    with pytest.raises(ValueError, match="manual.fillpad_expand: out tile rows/cols must be >= src tile rows/cols"):
        @pl.program
        class InvalidFillPadExpandShapeProgram:
            @pl.function
            def invalid_fillpad_expand_shape(
                self,
                output: pl.Tensor[[16, 8], pl.FP32],
            ):
                src_type = plm.TileType(
                    shape=[16, 16],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                )
                dst_type = plm.TileType(
                    shape=[16, 8],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                    pad=plm.TilePad.zero,
                )
                src = plm.make_tile(src_type, addr=0x0000, size=1024)
                dst = plm.make_tile(dst_type, addr=0x1000, size=512)
                plm.fillpad_expand(dst, src)
                return output


def test_pto_codegen_dump_tile_dynamic_offset_lowering():
    """dump_tile window lowering should accept dynamic offsets while keeping static sizes."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class DumpDynamicTileWindowProgram:
        @pl.function
        def dump_dynamic_tile_window(
            self,
            input: pl.Tensor[[32, 32], pl.FP32],
            row_off: pl.Scalar[pl.INDEX],
            output: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile = pl.load(input, offsets=[0, 0], shapes=[16, 16])
            plm.dump_tile(tile, offsets=[row_off, 0], shapes=[8, 16])
            pl.store(tile, offsets=[0, 0], shapes=[16, 16], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(DumpDynamicTileWindowProgram)

    codegen_obj = PTOCodegen()
    mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

    assert "pto.subset" in mlir_code
    assert "sizes [8, 16]" in mlir_code
    assert "v_row=?, v_col=16" in mlir_code


def test_pto_codegen_block_mul():
    """Test that block.mul generates pto.tmul."""

    @pl.program
    class MulProgram:
        @pl.function
        def mul_test(
            self,
            a: pl.Tensor[[32, 32], pl.FP32],
            b: pl.Tensor[[32, 32], pl.FP32],
            c: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 32])
            tile_c = pl.mul(tile_a, tile_b)
            pl.store(tile_c, offsets=[0, 0], shapes=[32, 32], output_tensor=c)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(MulProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify tmul generation
    assert "pto.tmul" in mlir_code
    assert "ins(" in mlir_code
    assert "outs(" in mlir_code


def test_pto_codegen_block_adds():
    """Test that block.adds generates pto.tadds with scalar constant."""

    @pl.program
    class AddsProgram:
        @pl.function
        def adds_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.add(tile_a, 3.14)
            pl.store(tile_b, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(AddsProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify tadds generation
    assert "pto.tadds" in mlir_code

    # Verify scalar constant generation
    assert "arith.constant" in mlir_code
    assert ": f32" in mlir_code


def test_pto_codegen_constants():
    """Test that constants are generated correctly."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class ConstantProgram:
        @pl.function
        def const_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile_a, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(ConstantProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify index constants
    assert "arith.constant" in mlir_code
    assert ": index" in mlir_code
    assert "%c0" in mlir_code or "%c32" in mlir_code


def test_pto_codegen_ssa_naming():
    """Test that SSA value names are correct."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class SSAProgram:
        @pl.function
        def ssa_test(
            self,
            a: pl.Tensor[[32, 32], pl.FP32],
            b: pl.Tensor[[32, 32], pl.FP32],
            c: pl.Tensor[[32, 32], pl.FP32],
        ):
            tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 32])
            tile_c = pl.mul(tile_a, tile_b)
            pl.store(tile_c, offsets=[0, 0], shapes=[32, 32], output_tensor=c)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(SSAProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify SSA value naming pattern
    assert "%arg0" in mlir_code  # Function parameters
    assert "%0" in mlir_code or "%1" in mlir_code  # Temporary values
    assert "%c" in mlir_code  # Constants


def test_pto_codegen_code_generation_order():
    """Test that code is generated in correct order: constants, views, allocs, body."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class OrderProgram:
        @pl.function
        def order_test(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(OrderProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    lines = mlir_code.split("\n")

    # Find indices of key operations
    const_idx = next((i for i, line in enumerate(lines) if "arith.constant" in line), -1)
    view_idx = next((i for i, line in enumerate(lines) if "make_tensor_view" in line), -1)
    alloc_idx = next((i for i, line in enumerate(lines) if "alloc_tile" in line), -1)
    load_idx = next((i for i, line in enumerate(lines) if "tload" in line), -1)

    # Verify order: constants < make_tensor_view < alloc_tile < operations
    assert const_idx < view_idx, "Constants should come before make_tensor_view"
    assert view_idx < alloc_idx, "make_tensor_view should come before alloc_tile"
    assert alloc_idx < load_idx, "alloc_tile should come before tload"


def test_pto_codegen_multiple_functions():
    """Test PTOCodegen with multiple functions."""

    @pl.program
    class MultiFunc:
        @pl.function
        def func1(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

        @pl.function
        def func2(self, x: pl.Tensor[[32, 32], pl.FP32], y: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(x, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=y)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(MultiFunc)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    # Verify both functions are present
    assert "func.func @func1" in mlir_code
    assert "func.func @func2" in mlir_code


def test_pto_codegen_reusability():
    """Test that the same PTOCodegen instance can be used multiple times."""

    @pl.program
    class ReusableProgram:
        @pl.function
        def test_func(self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]):
            tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
            pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=b)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(ReusableProgram)

    # Use the same codegen instance multiple times
    codegen = PTOCodegen()

    code1 = _get_mlir_code(codegen.generate(transformed_program))
    code2 = _get_mlir_code(codegen.generate(transformed_program))

    # Verify both calls produce valid code
    assert isinstance(code1, str)
    assert isinstance(code2, str)
    assert "func.func @test_func" in code1
    assert "func.func @test_func" in code2
    assert code1 == code2  # Should produce identical output


# --- Kernel wrapper generation tests ---


class TestPreprocessPtoasOutput:
    """Tests for _preprocess_ptoas_output."""

    def test_strips_include(self):
        result = _preprocess_ptoas_output(SAMPLE_PTOAS_OUTPUT)
        assert '#include "pto/pto-inst.hpp"' not in result

    def test_strips_using_namespace(self):
        result = _preprocess_ptoas_output(SAMPLE_PTOAS_OUTPUT)
        assert "using namespace pto;" not in result

    def test_replaces_global_aicore(self):
        result = _preprocess_ptoas_output(SAMPLE_PTOAS_OUTPUT)
        assert "__global__ AICORE void" not in result
        assert "static __aicore__ void test_func" in result

    def test_preserves_function_body(self):
        result = _preprocess_ptoas_output(SAMPLE_PTOAS_OUTPUT)
        assert "TLOAD(v1);" in result
        assert "TADDS(v2);" in result
        assert "TSTORE(v3);" in result

    def test_preserves_helpers(self):
        result = _preprocess_ptoas_output(SAMPLE_PTOAS_OUTPUT)
        assert "ptoas_bitcast" in result


class TestGenerateArgUnpacking:
    """Tests for _generate_arg_unpacking."""

    def test_tensor_only(self):
        func = _make_func("test_fn", [("a", "tensor"), ("b", "tensor"), ("out", "tensor")])
        code, names = _generate_arg_unpacking(func)
        assert "reinterpret_cast<__gm__ Tensor*>(args[0])" in code
        assert "reinterpret_cast<__gm__ Tensor*>(args[1])" in code
        assert "reinterpret_cast<__gm__ Tensor*>(args[2])" in code
        assert names == ["a", "b", "out"]

    def test_mixed_tensor_scalar(self):
        func = _make_func("test_fn", [("input", "tensor"), ("scale", "scalar"), ("output", "tensor")])
        code, names = _generate_arg_unpacking(func)
        assert "reinterpret_cast<__gm__ Tensor*>(args[0])" in code
        assert "scale_conv.u64 = args[1];" in code
        assert "float scale = scale_conv.val;" in code
        assert "reinterpret_cast<__gm__ Tensor*>(args[2])" in code
        assert names == ["input", "scale", "output"]

    def test_scalar_only(self):
        func = _make_func("test_fn", [("x", "scalar"), ("y", "scalar")])
        code, names = _generate_arg_unpacking(func)
        assert "x_conv.u64 = args[0];" in code
        assert "y_conv.u64 = args[1];" in code
        assert names == ["x", "y"]

    def test_dynamic_tensor_extracts_repeats_dims(self):
        func = _get_dyn_incore_func()
        code, names = _generate_arg_unpacking(func)
        # TH is dim 0 of first tensor a_0 — read from a_0_tensor->repeats[0]
        assert "a_0_tensor->repeats[0]" in code
        assert "int64_t TH" in code
        # TW is dim 1 of first tensor a_0 — read from a_0_tensor->repeats[1]
        assert "a_0_tensor->repeats[1]" in code
        assert "int64_t TW" in code
        # dynamic dims appended after tensor params
        assert names == ["a_0", "b_0", "output_0", "TH", "TW"]

    def test_dynamic_tensor_deduplicates_vars(self):
        # TH and TW each appear in a_0, b_0, and output_0 but should be extracted only once
        func = _get_dyn_incore_func()
        code, names = _generate_arg_unpacking(func)
        assert code.count("int64_t TH") == 1
        assert code.count("int64_t TW") == 1


class TestGenerateKernelWrapper:
    """Tests for _generate_kernel_wrapper."""

    def test_contains_kernel_entry(self):
        func = _make_func("my_kernel", [("a", "tensor"), ("s", "scalar"), ("out", "tensor")])
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        assert "void kernel_entry(__gm__ int64_t* args)" in wrapper

    def test_contains_includes(self):
        func = _make_func("my_kernel", [("a", "tensor"), ("s", "scalar"), ("out", "tensor")])
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        assert "#include <cstdint>" in wrapper
        assert "#include <pto/pto-inst.hpp>" in wrapper
        assert '#include "tensor.h"' in wrapper

    def test_contains_forward_call(self):
        func = _make_func("my_kernel", [("a", "tensor"), ("s", "scalar"), ("out", "tensor")])
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        assert "my_kernel(a, s, out);" in wrapper

    def test_ptoas_code_made_static(self):
        func = _make_func("my_kernel", [("a", "tensor"), ("s", "scalar"), ("out", "tensor")])
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        assert "__global__ AICORE" not in wrapper
        assert "static __aicore__ void test_func" in wrapper

    def test_no_duplicate_includes(self):
        func = _make_func("my_kernel", [("a", "tensor"), ("s", "scalar"), ("out", "tensor")])
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        count = wrapper.count("#include <pto/pto-inst.hpp>")
        assert count == 1, f"Expected 1 pto-inst include, found {count}"

    def test_dynamic_shape_forward_call_includes_dims(self):
        func = _get_dyn_incore_func()
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        # Forward call must include dynamic dims TH and TW after tensor args (SSA-renamed with _0 suffix)
        assert "dyn_func(a_0, b_0, output_0, TH, TW);" in wrapper

    def test_dynamic_shape_repeats_extraction_in_wrapper(self):
        func = _get_dyn_incore_func()
        wrapper = _generate_kernel_wrapper(func, SAMPLE_PTOAS_OUTPUT)
        assert "a_0_tensor->repeats[0]" in wrapper
        assert "a_0_tensor->repeats[1]" in wrapper


class TestGenerateSkipPtoas:
    """Tests for generate() with skip_ptoas=True."""

    def test_returns_pto_files(self, tmp_path):
        """When skip_ptoas=True, result keys for InCore functions end with .pto, not .cpp."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.PTO)

        @pl.program
        class SkipPtoasProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def skip_test(
                self, a: pl.Tensor[[32, 32], pl.FP32], b: pl.Tensor[[32, 32], pl.FP32]
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
                out = pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=b)
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
        transformed_program = pm.run_passes(SkipPtoasProgram)

        result = generate(transformed_program, str(tmp_path), skip_ptoas=True)

        kernel_keys = [k for k in result if k.startswith("kernels/")]
        assert len(kernel_keys) > 0, "Expected at least one kernel file"
        for key in kernel_keys:
            assert key.endswith(".pto"), f"Expected .pto extension, got: {key}"
            assert not key.endswith(".cpp"), f"Unexpected .cpp extension: {key}"


class TestMakeTensorCodegen:
    """Tests for pl.make_tensor body op generating pto.make_tensor_view in function body."""

    def test_make_tensor_emits_view_in_body(self):
        """pl.make_tensor in function body generates pto.make_tensor_view before return."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.PTO)

        @pl.program
        class MakeTensorProgram:
            @pl.function
            def make_tensor_func(self, a: pl.Ptr[pl.FP32]):
                view: pl.Tensor[[32, 32], pl.FP32] = pl.make_tensor(a, [32, 32], [32, 1])
                tile = pl.load(view, offsets=[0, 0], shapes=[32, 32])
                pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=view)

        pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
        transformed_program = pm.run_passes(MakeTensorProgram)

        codegen_obj = PTOCodegen()
        mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

        # pto.make_tensor_view must appear in the function body (after the header)
        assert "pto.make_tensor_view" in mlir_code
        # The body make_tensor_view should have the user-specified shape and stride
        assert "shape = [%c32, %c32]" in mlir_code
        assert "strides = [%c32, %c1]" in mlir_code
        # It must appear before "return"
        view_pos = mlir_code.find("pto.make_tensor_view")
        return_pos = mlir_code.rfind("return")
        assert view_pos < return_pos, "make_tensor_view must appear before return"


class TestTensorStrideCodegen:
    """Tests for pl.view(stride=[...]) annotation generating correct strides in make_tensor_view."""

    def test_tensor_param_with_custom_stride(self):
        """Tensor param with pl.view(stride=[...]) uses user-specified strides."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.PTO)

        @pl.program
        class CustomStrideProgram:
            @pl.function
            def custom_stride_func(self, a: pl.Tensor[[64, 64], pl.FP32, pl.view(stride=[128, 1])]):
                tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
                pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=a)

        pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
        transformed_program = pm.run_passes(CustomStrideProgram)

        codegen_obj = PTOCodegen()
        mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

        assert "shape = [%c64, %c64]" in mlir_code
        # User-specified stride [128, 1] must appear, NOT the default [64, 1]
        assert "strides = [%c128, %c1]" in mlir_code

    def test_tensor_param_without_stride_uses_default(self):
        """Tensor param without pl.view stride uses default row-major stride."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.PTO)

        @pl.program
        class DefaultStrideProgram:
            @pl.function
            def default_stride_func(self, a: pl.Tensor[[64, 64], pl.FP32]):
                tile = pl.load(a, offsets=[0, 0], shapes=[32, 32])
                pl.store(tile, offsets=[0, 0], shapes=[32, 32], output_tensor=a)

        pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
        transformed_program = pm.run_passes(DefaultStrideProgram)

        codegen_obj = PTOCodegen()
        mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

        assert "shape = [%c64, %c64]" in mlir_code
        # Default stride for [64, 64] is [64, 1]
        assert "strides = [%c64, %c1]" in mlir_code


class TestAddPtrCodegen:
    """Tests for pl.addptr generating pto.addptr in ptoas codegen."""

    def test_addptr_emits_pto_addptr(self):
        """pl.addptr generates pto.addptr and pl.make_tensor generates pto.make_tensor_view."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.PTO)

        @pl.program
        class AddPtrProgram:
            @pl.function
            def addptr_func(self, workspace: pl.Ptr[pl.FP32]):
                buf0: pl.Ptr[pl.FP32] = pl.addptr(workspace, 0)
                buf1: pl.Ptr[pl.FP32] = pl.addptr(buf0, 1024)
                view0: pl.Tensor[[32, 32], pl.FP32] = pl.make_tensor(buf0, [32, 32], [32, 1])
                view1: pl.Tensor[[32, 32], pl.FP32] = pl.make_tensor(buf1, [32, 32], [32, 1])

        pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
        transformed_program = pm.run_passes(AddPtrProgram)

        codegen_obj = PTOCodegen()
        mlir_code = _get_mlir_code(codegen_obj.generate(transformed_program))

        # pto.addptr must appear for each addptr call
        assert "pto.addptr" in mlir_code
        assert mlir_code.count("pto.addptr") == 2

        # pto.make_tensor_view must appear for each make_tensor call
        assert mlir_code.count("pto.make_tensor_view") == 2

        
def test_pto_codegen_section_vector():
    """Test that pl.section_vector() generates pto.section.vector { ... }."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class SectionVectorProgram:
        @pl.function
        def section_vector_test(
            self,
            input: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ):
            with pl.section_vector():
                tile = pl.load(input, offsets=[0, 0], shapes=[32, 32])
                result = pl.mul(tile, tile)
                pl.store(result, offsets=[0, 0], shapes=[32, 32], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(SectionVectorProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    assert "pto.section.vector {" in mlir_code
    assert "}" in mlir_code
    assert "pto.tload" in mlir_code
    assert "pto.tmul" in mlir_code
    assert "pto.tstore" in mlir_code


def test_pto_codegen_section_cube():
    """Test that pl.section_cube() generates pto.section.cube { ... }."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class SectionCubeProgram:
        @pl.function
        def section_cube_test(
            self,
            input: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ):
            with pl.section_cube():
                tile = pl.load(input, offsets=[0, 0], shapes=[32, 32])
                result = pl.add(tile, 1.0)
                pl.store(result, offsets=[0, 0], shapes=[32, 32], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(SectionCubeProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    assert "pto.section.cube {" in mlir_code
    assert "}" in mlir_code
    assert "pto.tload" in mlir_code
    assert "pto.tadds" in mlir_code
    assert "pto.tstore" in mlir_code


def test_pto_codegen_nested_sections():
    """Test nested section_vector and section_cube."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class NestedSectionProgram:
        @pl.function
        def nested_section_test(
            self,
            a: pl.Tensor[[32, 32], pl.FP32],
            b: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ):
            with pl.section_vector():
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
                result_a = pl.mul(tile_a, tile_a)
                pl.store(result_a, offsets=[0, 0], shapes=[32, 32], output_tensor=output)

            with pl.section_cube():
                tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 32])
                result_b = pl.add(tile_b, 2.0)
                pl.store(result_b, offsets=[0, 0], shapes=[32, 32], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(NestedSectionProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    assert mlir_code.count("pto.section.vector {") == 1
    assert mlir_code.count("pto.section.cube {") == 1


def test_pto_codegen_section_with_for_loop():
    """Test section_vector with for loop inside."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)

    @pl.program
    class SectionForLoopProgram:
        @pl.function
        def section_for_test(
            self,
            input: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ):
            with pl.section_vector():
                for i in pl.range(0, 2):
                    tile = pl.load(input, offsets=[i * 32, 0], shapes=[32, 32])
                    result = pl.add(tile, 1.0)
                    pl.store(result, offsets=[i * 32, 0], shapes=[32, 32], output_tensor=output)

    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(SectionForLoopProgram)

    codegen = PTOCodegen()
    mlir_code = _get_mlir_code(codegen.generate(transformed_program))

    assert "pto.section.vector {" in mlir_code
    assert "scf.for" in mlir_code
    assert "pto.tload" in mlir_code
    assert "pto.tadds" in mlir_code
    assert "pto.tstore" in mlir_code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
