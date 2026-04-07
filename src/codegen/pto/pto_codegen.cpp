/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/ir/transforms/printer.h"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using ir::As;
using ir::AssignStmtPtr;
using ir::BinaryExprPtr;
using ir::CallPtr;
using ir::EvalStmtPtr;
using ir::ExprPtr;
using ir::ForStmtPtr;
using ir::FunctionPtr;
using ir::IfStmtPtr;
using ir::MemRefPtr;
using ir::OpStmtsPtr;
using ir::ProgramPtr;
using ir::PtrType;
using ir::ScalarType;
using ir::SeqStmtsPtr;
using ir::StmtPtr;
using ir::TensorType;
using ir::TileType;
using ir::VarPtr;
using ir::YieldStmtPtr;

// Helper function to convert DataType to MLIR type string
static std::string DataTypeToMLIRImpl(::pypto::DataType dtype) {
  if (dtype == ::pypto::DataType::FP32) {
    return "f32";
  } else if (dtype == ::pypto::DataType::FP16) {
    return "f16";
  } else if (dtype == ::pypto::DataType::BF16) {
    return "bf16";
  } else if (dtype == ::pypto::DataType::INT8) {
    return "i8";
  } else if (dtype == ::pypto::DataType::UINT8) {
    return "ui8";
  } else if (dtype == ::pypto::DataType::INT16) {
    return "i16";
  } else if (dtype == ::pypto::DataType::UINT16) {
    return "ui16";
  } else if (dtype == ::pypto::DataType::INT32) {
    return "i32";
  } else if (dtype == ::pypto::DataType::UINT32) {
    return "ui32";
  } else if (dtype == ::pypto::DataType::INDEX) {
    return "index";
  } else if (dtype == ::pypto::DataType::INT64) {
    return "i64";
  } else if (dtype == ::pypto::DataType::UINT64) {
    return "ui64";
  } else if (dtype == ::pypto::DataType::BOOL) {
    return "i1";
  } else {
    throw pypto::ValueError("Invalid DataType value");
  }
}

// Helper function to convert MemorySpace to PTO address space string
static std::string MemorySpaceToMLIR(ir::MemorySpace space) {
  if (space == ir::MemorySpace::DDR) {
    return "gm";
  } else if (space == ir::MemorySpace::Vec) {
    return "vec";
  } else if (space == ir::MemorySpace::Mat) {
    return "mat";
  } else if (space == ir::MemorySpace::Left) {
    return "left";
  } else if (space == ir::MemorySpace::Right) {
    return "right";
  } else if (space == ir::MemorySpace::Acc) {
    return "acc";
  } else {
    throw pypto::ValueError("Invalid MemorySpace value");
  }
}

// Visitor to collect all MemRef objects from TileType variables
class MemRefCollectorVisitor : public ir::IRVisitor {
 public:
  MemRefCollectorVisitor() = default;

  [[nodiscard]] const std::vector<MemRefPtr>& GetMemRefs() const { return memrefs_; }
  [[nodiscard]] const std::map<const ir::MemRef*, std::shared_ptr<const TileType>>& GetMemRefTileTypes()
      const {
    return memref_tile_types_;
  }

  /// Returns true if the given memref was first seen inside a Cube section.
  [[nodiscard]] bool IsCubeOnly(const ir::MemRef* m) const { return cube_memrefs_.count(m) > 0; }
  /// Returns true if the given memref was first seen inside a Vector section.
  [[nodiscard]] bool IsVecOnly(const ir::MemRef* m) const { return vec_memrefs_.count(m) > 0; }

  void VisitExpr_(const VarPtr& op) override {
    auto tile_type = As<TileType>(op->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      AddMemRefIfUnique(tile_type->memref_.value(), tile_type);
    }
  }

  void VisitExpr_(const ir::IterArgPtr& op) override {
    auto tile_type = As<TileType>(op->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      AddMemRefIfUnique(tile_type->memref_.value(), tile_type);
    }
  }

  void VisitStmt_(const ir::SectionStmtPtr& op) override {
    auto prev = current_section_;
    current_section_ = op->section_kind_;
    IRVisitor::VisitStmt_(op);
    current_section_ = prev;
  }

 private:
  std::vector<MemRefPtr> memrefs_;
  std::set<const ir::MemRef*> seen_ptrs_;
  std::map<const ir::MemRef*, std::shared_ptr<const TileType>> memref_tile_types_;
  std::set<const ir::MemRef*> cube_memrefs_;
  std::set<const ir::MemRef*> vec_memrefs_;
  std::optional<ir::SectionKind> current_section_;

  void AddMemRefIfUnique(const MemRefPtr& memref, const std::shared_ptr<const TileType>& tile_type) {
    const ir::MemRef* raw_ptr = memref.get();
    if (seen_ptrs_.find(raw_ptr) == seen_ptrs_.end()) {
      memrefs_.push_back(memref);
      seen_ptrs_.insert(raw_ptr);
      memref_tile_types_[raw_ptr] = tile_type;
      // Track section membership
      if (current_section_ == ir::SectionKind::Cube) cube_memrefs_.insert(raw_ptr);
      if (current_section_ == ir::SectionKind::Vector) vec_memrefs_.insert(raw_ptr);
    }
  }
};

// ========================================================================
// Constructors
// ========================================================================

PTOCodegen::PTOCodegen() : backend_(backend::GetBackend()) {
  auto type = backend::GetBackendType();
  CHECK(type == backend::BackendType::PTO)
      << "PTOCodegen requires PTO backend, but " << (type == backend::BackendType::CCE ? "CCE" : "unknown")
      << " is configured";
}

PTOCodegen::PTOCodegen(const backend::Backend* backend) : backend_(backend) {
  CHECK(backend != nullptr) << "Backend cannot be null";
}

// ========================================================================
// Generate entry and GenerateFunction
// ========================================================================
std::string PTOCodegen::Generate(const ProgramPtr& program) {
  // Lower break/continue to structured control flow before emitting MLIR.
  // This pass is idempotent: programs without break/continue are unchanged.
  // Must run BEFORE ConvertToSSA: continue/break lowering restructures the loop
  // body (e.g. wrapping remaining stmts in an else-branch) while variables are
  // still in non-SSA form. ConvertToSSA then correctly produces iter_args and
  // scf.yield for the restructured body.
  ir::ProgramPtr lowered = ir::pass::LowerBreakContinue()(program);

  // Convert to SSA form so that loop-carried variables (e.g. `loop = loop + 1`
  // inside a while body) become proper scf.while iter_args with scf.yield.
  ir::ProgramPtr ssa_program = ir::pass::ConvertToSSA()(lowered);

  // Fold constants and simplify if-stmts to reduce scalar ops in generated code.
  ir::ProgramPtr opt_program = ir::pass::ConstFoldAndSimplify()(ssa_program);

  stream_.str("");
  stream_.clear();
  constants_section_.str("");
  constants_section_.clear();
  body_section_.str("");
  body_section_.clear();

  indent_level_ = 0;
  function_body_indent_level_ = 0;

  stream_ << "module {\n";
  indent_level_++;

  for (const auto& [gvar, func] : opt_program->functions_) {
    if (func->func_type_ == ir::FunctionType::Orchestration) {
      throw pypto::ValueError(
          "PTO backend does not support Orchestration functions. "
          "Function '" +
          func->name_ + "' is marked as Orchestration. ");
    }
    GenerateFunction(func);
  }

  indent_level_--;
  stream_ << "}\n";
  return stream_.str();
}

void PTOCodegen::GenerateFunction(const FunctionPtr& func) {
  if (func->func_type_ == ir::FunctionType::Helper) {
    GenerateHelperFunction(func);
    return;
  }
  current_function_ = func;
  temp_counter_ = 0;
  last_assigned_temp_ = "";
  var_to_mlir_.clear();
  tensor_to_view_.clear();
  memref_to_mlir_.clear();
  var_to_memref_.clear();
  memref_to_tile_type_.clear();
  emitted_constants_.clear();
  emitted_float_constants_.clear();
  emitted_i64_constants_.clear();
  float_const_names_.clear();
  extra_alloc_tiles_.clear();
  extra_tile_buf_types_.clear();
  tuple_var_to_make_tuple_.clear();
  tile_valid_shapes_.clear();
  indirect_select_depth_ = 0;
  indirect_addr_vars_.clear();
  constants_section_.str("");
  constants_section_.clear();
  body_section_.str("");
  body_section_.clear();
  function_body_indent_level_ = 0;

  BuildVarToMemRefMapping(func);

  MemRefCollectorVisitor collector;
  if (func->body_) {
    collector.VisitStmt(func->body_);
  }

  for (const auto& memref : collector.GetMemRefs()) {
    std::string tile_buf = NewTemp();
    memref_to_mlir_[memref.get()] = tile_buf;
  }
  memref_to_tile_type_ = collector.GetMemRefTileTypes();
  // Track section-specific memrefs for deferred emission inside sections
  vec_only_memrefs_.clear();
  cube_only_memrefs_.clear();
  all_memrefs_ = collector.GetMemRefs();
  for (const auto& memref : all_memrefs_) {
    if (collector.IsVecOnly(memref.get())) vec_only_memrefs_.insert(memref.get());
    if (collector.IsCubeOnly(memref.get())) cube_only_memrefs_.insert(memref.get());
  }

  // Collect ordered unique dynamic dimension variables from tensor parameter shapes
  std::vector<std::string> dyn_var_names;
  {
    std::set<std::string> seen_dyn_vars;
    for (const auto& param : func->params_) {
      if (auto tensor_type = As<TensorType>(param->GetType())) {
        for (const auto& dim : tensor_type->shape_) {
          if (auto var = As<ir::Var>(dim)) {
            if (seen_dyn_vars.find(var->name_) == seen_dyn_vars.end()) {
              dyn_var_names.push_back(var->name_);
              seen_dyn_vars.insert(var->name_);
            }
          }
        }
      }
    }
  }

  stream_ << GetIndent() << "func.func @" << func->name_ << "(";

  std::set<std::string> param_names;
  for (size_t i = 0; i < func->params_.size(); i++) {
    if (i > 0) stream_ << ", ";
    const auto& param = func->params_[i];
    std::string arg_name = "%arg" + std::to_string(i);
    stream_ << arg_name << ": ";

    var_to_mlir_[param->name_] = arg_name;
    param_names.insert(param->name_);

    if (auto tensor_type = As<TensorType>(param->GetType())) {
      stream_ << "!pto.ptr<" << GetTypeString(tensor_type->dtype_) << ">";
    } else if (auto ptr_type = As<PtrType>(param->GetType())) {
      // PtrType params are raw pointers: emit as !pto.ptr<dtype>, no preamble view needed
      stream_ << "!pto.ptr<" << GetTypeString(ptr_type->dtype_) << ">";
    } else if (auto scalar_type = As<ScalarType>(param->GetType())) {
      stream_ << GetTypeString(scalar_type->dtype_);
    } else {
      stream_ << "!pto.ptr<f32>";
    }
  }

  // Append trailing index parameters for each unique dynamic dimension variable
  size_t next_arg_idx = func->params_.size();
  for (const auto& var_name : dyn_var_names) {
    std::string arg_name = "%arg" + std::to_string(next_arg_idx++);
    stream_ << ", " << arg_name << ": index";
    var_to_mlir_[var_name] = arg_name;
  }

  stream_ << ") {\n";
  indent_level_++;
  function_body_indent_level_ = indent_level_;

  for (const auto& [var_name, memref_ptr] : var_to_memref_) {
    if (param_names.find(var_name) == param_names.end()) {
      var_to_mlir_[var_name] = memref_to_mlir_[memref_ptr];
    }
  }

  for (const auto& var : func->params_) {
    if (auto tensor_type = As<TensorType>(var->GetType())) {
      std::string tensor_view = NewTemp();
      tensor_to_view_[var->name_] = tensor_view;
      tensor_to_ptr_[var->name_] = var_to_mlir_.at(var->name_);  // raw %argN

      for (const auto& j : tensor_type->shape_) {
        if (As<ir::ConstInt>(j)) {
          GetOrEmitIndexConstant(GetConstIntValue(j));
        }
      }
      if (tensor_type->shape_.size() == 2) {
        if (As<ir::ConstInt>(tensor_type->shape_[1])) {
          GetOrEmitIndexConstant(GetConstIntValue(tensor_type->shape_[1]));
        }
        GetOrEmitIndexConstant(1);
      } else if (tensor_type->shape_.size() == 1) {
        GetOrEmitIndexConstant(1);
      }
    }
  }

  auto saved_stream = std::move(stream_);

  // Emit Tiles and Tensors create
  std::ostringstream alloc_tiles_buffer;
  stream_ = std::move(alloc_tiles_buffer);
  EmitMakeTensorViews(func);
  EmitAllocTiles(func, collector.GetMemRefs());
  EmitExtraAllocTiles();
  std::string alloc_tiles_content = stream_.str();

  stream_ = std::move(body_section_);

  if (func->body_) {
    VisitStmt(func->body_);
  }

  std::string body_content = stream_.str();
  stream_ = std::move(saved_stream);

  // Step 1: Output constants_section_ FIRST (includes i64 constants for addr operands)
  stream_ << constants_section_.str();
  // Step 2: Then output alloc_tiles (which references the i64 constants)
  stream_ << alloc_tiles_content;
  // Step 3: Finally output the body
  stream_ << body_content;
  stream_ << GetIndent() << "return\n";

  indent_level_--;
  stream_ << GetIndent() << "}\n";
  function_body_indent_level_ = 0;
}

void PTOCodegen::BuildVarToMemRefMapping(const FunctionPtr& func) {
  class VarMemRefMapper : public ir::IRVisitor {
   public:
    std::map<std::string, const ir::MemRef*>& var_to_memref;

    explicit VarMemRefMapper(std::map<std::string, const ir::MemRef*>& mapping) : var_to_memref(mapping) {}

    void VisitStmt_(const AssignStmtPtr& op) override {
      if (auto tile_type = As<TileType>(op->var_->GetType())) {
        if (tile_type->memref_.has_value()) {
          var_to_memref[op->var_->name_] = tile_type->memref_.value().get();
        }
      }
      ir::IRVisitor::VisitStmt_(op);
    }
  };

  VarMemRefMapper mapper(var_to_memref_);
  if (func->body_) {
    mapper.VisitStmt(func->body_);
  }
}

void PTOCodegen::EmitMakeTensorViews(const FunctionPtr& func) {
  for (size_t i = 0; i < func->params_.size(); i++) {
    const auto& param = func->params_[i];
    if (auto tensor_type = As<TensorType>(param->GetType())) {
      std::string tensor_view = tensor_to_view_[param->name_];

      // Pre-compute strides for multi-dimensional tensors (rank >= 3)
      // For shape [d0, d1, ..., dn-1], strides = [d1*d2*...*dn-1, d2*...*dn-1, ..., dn-1, 1]
      std::vector<std::string> stride_values;
      if (!tensor_type->tensor_view_.has_value() || tensor_type->tensor_view_->stride.empty()) {
        if (tensor_type->shape_.size() > 2) {
          // Compute strides for each dimension
          for (size_t j = 0; j < tensor_type->shape_.size(); j++) {
            if (j == tensor_type->shape_.size() - 1) {
              // Last dimension: stride = 1
              stride_values.push_back(GetOrEmitIndexConstant(1));
            } else {
              // Compute product of remaining dimensions
              std::string result;
              for (size_t k = j + 1; k < tensor_type->shape_.size(); k++) {
                std::string dim_val;
                if (auto var = As<ir::Var>(tensor_type->shape_[k])) {
                  dim_val = var_to_mlir_.at(var->name_);
                } else {
                  dim_val = GetOrEmitIndexConstant(GetConstIntValue(tensor_type->shape_[k]));
                }
                if (k == j + 1) {
                  result = dim_val;
                } else {
                  std::string product = NewTemp();
                  stream_ << GetIndent() << product << " = arith.muli " << result << ", " << dim_val << " : index\n";
                  result = product;
                }
              }
              stride_values.push_back(result);
            }
          }
        }
      }

      stream_ << GetIndent() << tensor_view << " = pto.make_tensor_view ";
      stream_ << "%arg" << i;

      stream_ << ", shape = [";
      for (size_t j = 0; j < tensor_type->shape_.size(); j++) {
        if (j > 0) stream_ << ", ";
        if (auto var = As<ir::Var>(tensor_type->shape_[j])) {
          stream_ << var_to_mlir_.at(var->name_);
        } else {
          stream_ << GetOrEmitIndexConstant(GetConstIntValue(tensor_type->shape_[j]));
        }
      }
      stream_ << "],";

      stream_ << " strides = [";
      const bool has_custom_stride = tensor_type->tensor_view_.has_value() &&
                                      !tensor_type->tensor_view_->stride.empty();
      if (has_custom_stride) {
        // Use user-specified strides from pl.view(stride=[...])
        const auto& stride = tensor_type->tensor_view_->stride;
        for (size_t j = 0; j < stride.size(); j++) {
          if (j > 0) stream_ << ", ";
          if (auto var = As<ir::Var>(stride[j])) {
            stream_ << var_to_mlir_.at(var->name_);
          } else {
            stream_ << GetOrEmitIndexConstant(GetConstIntValue(stride[j]));
          }
        }
      } else {
        // Default: row-major stride derived from shape
        if (tensor_type->shape_.size() > 2) {
          // Use pre-computed strides
          for (size_t j = 0; j < stride_values.size(); j++) {
            if (j > 0) stream_ << ", ";
            stream_ << stride_values[j];
          }
        } else if (tensor_type->shape_.size() == 2) {
          if (auto var = As<ir::Var>(tensor_type->shape_[1])) {
            stream_ << var_to_mlir_.at(var->name_);
          } else {
            stream_ << GetOrEmitIndexConstant(GetConstIntValue(tensor_type->shape_[1]));
          }
          stream_ << ", " << GetOrEmitIndexConstant(1);
        } else if (tensor_type->shape_.size() == 1) {
          stream_ << GetOrEmitIndexConstant(1);
        }
      }
      stream_ << "]";

      stream_ << " : !pto.tensor_view<";
      for (size_t j = 0; j < tensor_type->shape_.size(); j++) {
        if (j > 0) stream_ << "x";
        stream_ << "?";
      }
      stream_ << "x" << GetTypeString(tensor_type->dtype_) << ">\n";
    }
  }
}

void PTOCodegen::EmitAllocTiles(const ir::FunctionPtr& func, const std::vector<ir::MemRefPtr>& memrefs,
                                 const std::set<const ir::MemRef*>* only_these) {
  (void)func;
  for (const auto& memref : memrefs) {
    // If only_these is specified, skip memrefs not in the set
    if (only_these && only_these->find(memref.get()) == only_these->end()) continue;
    // If emitting at function top (only_these is null), skip section-specific memrefs
    if (!only_these && (vec_only_memrefs_.count(memref.get()) || cube_only_memrefs_.count(memref.get()))) continue;
    std::string tile_buf = memref_to_mlir_[memref.get()];

    // Collect dynamic valid_shape variable names if present
    std::string valid_row_mlir;
    std::string valid_col_mlir;
    auto tile_it = memref_to_tile_type_.find(memref.get());
    if (tile_it != memref_to_tile_type_.end()) {
      const auto& tile_type = tile_it->second;
      if (tile_type->tile_view_.has_value()) {
        const auto& tv = tile_type->tile_view_.value();
        if (tv.valid_shape.size() >= 1) {
          if (auto var = As<ir::Var>(tv.valid_shape[0])) {
            valid_row_mlir = GetVarName(var);
          } else if (auto var = As<ir::ConstInt>(tv.valid_shape[0])) {
            if (var->value_ == -1) {
              valid_row_mlir = GetExprAsCode(tile_type->shape_[0]);
            }
          }
        }
        if (tv.valid_shape.size() >= 2) {
          if (auto var = As<ir::Var>(tv.valid_shape[1])) {
            valid_col_mlir = GetVarName(var);
          } else if (auto var = As<ir::ConstInt>(tv.valid_shape[1])) {
            if (var->value_ == -1) {
              valid_col_mlir = GetExprAsCode(tile_type->shape_[1]);
            }
          }
        }
      }
    }

    std::ostringstream line;
    line << tile_buf << " = pto.alloc_tile";
    // For level3, all alloc_tile must have addr operand.
    // If no explicit address is set, use 0 (compiler will assign).
    std::string addr_operand;
    if (memref->addr_) {
      if (auto const_addr = As<ir::ConstInt>(memref->addr_)) {
        addr_operand = GetOrEmitI64Constant(const_addr->value_);
      }
    }
    if (addr_operand.empty()) {
      addr_operand = GetOrEmitI64Constant(0);
    }
    line << " addr = " << addr_operand;
    if (!valid_row_mlir.empty()) line << " valid_row = " << valid_row_mlir;
    if (!valid_col_mlir.empty()) line << " valid_col = " << valid_col_mlir;
    line << " : " << GetTileBufTypeString(memref.get());
    stream_ << GetIndent() << line.str() << "\n";

    // Record tile → addr SSA mapping for ops that need to adjust tile address (e.g., tinsert with offset)
    tile_to_addr_ssa_[tile_buf] = addr_operand;

    // Initialize tile→valid_shape mapping for dynamic tiles
    if (!valid_row_mlir.empty() && !valid_col_mlir.empty()) {
      UpdateTileValidShape(tile_buf, valid_row_mlir, valid_col_mlir);
    }
  }
}

// ========================================================================
// Private helpers
// ========================================================================

std::string PTOCodegen::GetIndent() const { return std::string(static_cast<size_t>(indent_level_) * 2, ' '); }

std::string PTOCodegen::GetOrEmitIndexConstant(int64_t value) {
  std::string name = "%c" + std::to_string(value);
  if (emitted_constants_.find(value) == emitted_constants_.end()) {
    const int constants_indent_level = function_body_indent_level_ > 0 ? function_body_indent_level_ : indent_level_;
    constants_section_ << std::string(static_cast<size_t>(constants_indent_level) * 2, ' ')
                       << name << " = arith.constant " << value << " : index\n";
    emitted_constants_.insert(value);
  }
  return name;
}

std::string PTOCodegen::GetOrEmitI64Constant(int64_t value) {
  std::string name = "%addr" + std::to_string(value);
  if (emitted_i64_constants_.find(value) == emitted_i64_constants_.end()) {
    constants_section_ << GetIndent() << name << " = arith.constant " << value << " : i64\n";
    emitted_i64_constants_.insert(value);
  }
  return name;
}

std::string PTOCodegen::GetTileBufForMemRef(const MemRefPtr& memref) {
  auto it = memref_to_mlir_.find(memref.get());
  INTERNAL_CHECK(it != memref_to_mlir_.end()) << "MemRef not found in mapping";
  return it->second;
}

std::string PTOCodegen::AllocNewTileBuf(const std::string& tile_buf_type_string) {
  std::string name = NewTemp();
  extra_alloc_tiles_.emplace_back(name, tile_buf_type_string);
  extra_tile_buf_types_[name] = tile_buf_type_string;
  return name;
}

void PTOCodegen::SetCurrentResultBuf(const std::string& buf) { current_result_buf_ = buf; }

void PTOCodegen::EmitExtraAllocTiles() {
  for (const auto& [name, type_str] : extra_alloc_tiles_) {
    stream_ << GetIndent() << name << " = pto.alloc_tile : " << type_str << "\n";
  }
}

// ========================================================================
// Statement visitors
// ========================================================================

void PTOCodegen::VisitStmt_(const AssignStmtPtr& op) {
  if (auto call = As<ir::Call>(op->value_)) {
    if (backend_ != nullptr && backend_->GetOpInfo(call->op_->name_) != nullptr) {
      std::string result_buf = op->var_->name_;  // use for var_name to mlir name mapping for non-tile op
      std::shared_ptr<const TileType> result_tile_type;
      if (auto tile_type = As<TileType>(op->var_->GetType())) {
        if (tile_type->memref_.has_value()) {
          result_buf = GetTileBufForMemRef(tile_type->memref_.value());
        }
        result_tile_type = tile_type;
      }
      current_result_buf_ = result_buf;
      current_result_tile_type_ = result_tile_type;
      current_result_var_name_ = op->var_->name_;
      VisitExpr(op->value_);
      // Built-in tile-producing ops often write directly to current_result_buf_
      // without producing a separate SSA value in current_expr_value_. Record the
      // assigned buffer name so later uses like block.print(tile) can resolve it.
      if (result_tile_type && !current_result_buf_.empty()) {
        var_to_mlir_[op->var_->name_] = current_result_buf_;
      }
      // If codegen changed the result buffer (e.g., reshape allocated a new tile),
      // update variable mapping so subsequent references use the new buffer
      if (!current_result_buf_.empty() && current_result_buf_ != result_buf) {
        var_to_mlir_[op->var_->name_] = current_result_buf_;
      }

      current_result_var_name_.clear();
      current_result_buf_.clear();
      current_result_tile_type_ = nullptr;
      return;
    }
  }

  current_expr_value_ = "";
  // MakeTuple has no MLIR equivalent — store for TupleGetItemExpr resolution
  if (auto make_tuple = As<ir::MakeTuple>(op->value_)) {
    tuple_var_to_make_tuple_[op->var_->name_] = make_tuple;
    return;
  }
  // TupleGetItemExpr that resolves to a nested MakeTuple (e.g. from inlined function
  // returning tuple-of-tuples): propagate the inner MakeTuple to the new var name.
  if (auto tgi = As<ir::TupleGetItemExpr>(op->value_)) {
    if (auto tuple_var = As<ir::Var>(tgi->tuple_)) {
      auto it = tuple_var_to_make_tuple_.find(tuple_var->name_);
      if (it != tuple_var_to_make_tuple_.end()) {
        const auto& elems = it->second->elements_;
        if (tgi->index_ >= 0 && tgi->index_ < static_cast<int>(elems.size())) {
          auto elem = elems[tgi->index_];
          // Element is a Var pointing to another MakeTuple — propagate registration
          if (auto elem_var = As<ir::Var>(elem)) {
            auto inner_it = tuple_var_to_make_tuple_.find(elem_var->name_);
            if (inner_it != tuple_var_to_make_tuple_.end()) {
              tuple_var_to_make_tuple_[op->var_->name_] = inner_it->second;
              return;
            }
          }
          // Element is a MakeTuple directly
          if (auto inner_mt = As<ir::MakeTuple>(elem)) {
            tuple_var_to_make_tuple_[op->var_->name_] = inner_mt;
            return;
          }
        }
      }
    }
  }
  VisitExpr(op->value_);
  // mapping arith var name to mlir mapping
  if (!current_expr_value_.empty()) {
    var_to_mlir_[op->var_->name_] = current_expr_value_;
    // If the rhs is a tensor view (i.e. the assigned variable has TensorType),
    // propagate tensor_to_view_ and tensor_to_ptr_ so that later
    // GetOrCreateTensorView() and GetTensorPtr() calls on this lhs variable succeed.
    if (As<TensorType>(op->var_->GetType())) {
      tensor_to_view_[op->var_->name_] = current_expr_value_;
      // Propagate raw pointer: find which tensor's view matches current_expr_value_
      for (const auto& [name, ptr] : tensor_to_ptr_) {
        if (tensor_to_view_.count(name) && tensor_to_view_[name] == current_expr_value_) {
          tensor_to_ptr_[op->var_->name_] = ptr;
          break;
        }
      }
    }
    current_expr_value_ = "";
  }
}

// ========================================================================
// Expression visitors
// ========================================================================

void PTOCodegen::VisitExpr_(const CallPtr& op) {
  const std::string& op_name = op->op_->name_;

  CHECK(backend_ != nullptr) << "Backend must not be null; use PTOCodegen(backend) or default backend";
  const auto* op_info = backend_->GetOpInfo(op_name);
  if (op_info == nullptr) {
    // Not a built-in op — treat as a user-defined function call (func.call)
    EmitFuncCall(op);
    return;
  }

  last_assigned_temp_ = "";
  std::string mlir_line = op_info->codegen_func(op, *this);
  if (!mlir_line.empty()) {
    Emit(mlir_line);
  }
  // Always propagate the SSA result name allocated by the op codegen function.
  // This ensures statement-ops (e.g. get_block_idx) work correctly as sub-expressions
  // in binary operations where current_expr_value_ may hold a stale LHS value.
  if (!last_assigned_temp_.empty()) {
    current_expr_value_ = last_assigned_temp_;
  }
}

// ========================================================================
// CodegenBase interface and PTO-specific helper methods
// ========================================================================

std::string PTOCodegen::GetCurrentResultTarget() const { return current_result_buf_; }

void PTOCodegen::Emit(const std::string& line) { stream_ << GetIndent() << line << "\n"; }

std::string PTOCodegen::GetExprAsCode(const ExprPtr& expr) {
  if (auto var = As<ir::Var>(expr)) {
    return GetVarName(var);
  }
  if (auto const_int = As<ir::ConstInt>(expr)) {
    return GetIndexConstant(const_int->value_);
  }
  if (auto const_float = As<ir::ConstFloat>(expr)) {
    return GetOrEmitFloatConstant(const_float->value_, "f32");
  }

  // Fall back to visitor pattern for complex expressions (arithmetic, comparisons)
  current_expr_value_ = "";
  VisitExpr(expr);
  std::string result = current_expr_value_;
  current_expr_value_ = "";
  if (!result.empty()) {
    return result;
  }

  LOG_ERROR << "GetExprAsCode for unsupported expression type";
  return "";
}

std::string PTOCodegen::GetTypeString(const DataType& dtype) const { return DataTypeToMLIRImpl(dtype); }

std::string PTOCodegen::GetVarName(const VarPtr& var) {
  auto it = var_to_mlir_.find(var->name_);
  if (it != var_to_mlir_.end()) {
    return it->second;
  }
  auto memref_it = var_to_memref_.find(var->name_);
  if (memref_it != var_to_memref_.end()) {
    auto mlir_it = memref_to_mlir_.find(memref_it->second);
    if (mlir_it != memref_to_mlir_.end()) {
      return mlir_it->second;
    }
  }
  LOG_ERROR << "Variable " << var->name_ << " not found in MLIR mapping";
  return "";
}

std::string PTOCodegen::NewTemp() {
  std::string name = "%" + std::to_string(temp_counter_++);
  last_assigned_temp_ = name;
  return name;
}

void PTOCodegen::RegisterVarToMlir(const std::string& var_name, const std::string& mlir_name) {
  var_to_mlir_[var_name] = mlir_name;
}

int64_t PTOCodegen::GetConstIntValue(const ExprPtr& expr) {
  if (auto const_int = As<ir::ConstInt>(expr)) {
    return const_int->value_;
  }
  LOG_ERROR << "Expected ConstInt expression";
  return 0;
}

std::string PTOCodegen::GetOrCreateTensorView(const VarPtr& tensor_param) {
  auto it = tensor_to_view_.find(tensor_param->name_);
  INTERNAL_CHECK(it != tensor_to_view_.end())
      << "Tensor view not found for parameter: " << tensor_param->name_;
  return it->second;
}

std::string PTOCodegen::GetTensorPtr(const VarPtr& tensor_param) {
  auto it = tensor_to_ptr_.find(tensor_param->name_);
  INTERNAL_CHECK(it != tensor_to_ptr_.end())
      << "Tensor pointer not found for parameter: " << tensor_param->name_;
  return it->second;
}

std::string PTOCodegen::GetIndexConstant(int64_t val) { return GetOrEmitIndexConstant(val); }

std::string PTOCodegen::GetAddrConstant(int64_t value) { return GetOrEmitI64Constant(value); }

std::string PTOCodegen::GetOrEmitFloatConstant(double value, const std::string& mlir_type) {
  if (emitted_float_constants_.find(value) == emitted_float_constants_.end()) {
    std::string name = "%cst";
    if (!emitted_float_constants_.empty()) {
      name += "_" + std::to_string(emitted_float_constants_.size());
    }

    std::ostringstream val_str;
    val_str << std::scientific << std::setprecision(6) << value;

    const int constants_indent_level = function_body_indent_level_ > 0 ? function_body_indent_level_ : indent_level_;
    constants_section_ << std::string(static_cast<size_t>(constants_indent_level) * 2, ' ')
                       << name << " = arith.constant " << val_str.str() << " : " << mlir_type << "\n";
    emitted_float_constants_.insert(value);
    float_const_names_[value] = name;
    return name;
  }
  return float_const_names_[value];
}

std::string PTOCodegen::GetTensorViewTypeString(const ir::TensorType* tensor_type) const {
  std::ostringstream oss;
  oss << "!pto.tensor_view<";
  for (size_t i = 0; i < tensor_type->shape_.size(); i++) {
    if (i > 0) oss << "x";
    oss << "?";
  }
  oss << "x" << GetTypeString(tensor_type->dtype_) << ">";
  return oss.str();
}

// Helper to convert TileLayout to string
static const char* TileLayoutToStr(ir::TileLayout layout) {
  switch (layout) {
    case ir::TileLayout::none_box:
      return "none_box";
    case ir::TileLayout::row_major:
      return "row_major";
    case ir::TileLayout::col_major:
      return "col_major";
    default:
      INTERNAL_CHECK(false) << "Unknown TileLayout: " << static_cast<int>(layout);
      return "";  // Should be unreachable
  }
}

// Helper to format tile_buf type string from components
static std::string FormatTileBufTypeString(const std::string& loc, const std::string& dtype_str, int64_t rows,
                                           int64_t cols, ir::TileLayout blayout, ir::TileLayout slayout,
                                           uint64_t fractal, ir::TilePad pad, int64_t v_row, int64_t v_col,
                                           bool v_row_dynamic = false, bool v_col_dynamic = false,
                                           ir::CompactMode compact = ir::CompactMode::null) {
  std::ostringstream oss;
  oss << "!pto.tile_buf<loc=" << loc << ", dtype=" << dtype_str;
  oss << ", rows=" << rows << ", cols=" << cols;
  oss << ", v_row=" << (v_row_dynamic ? "?" : std::to_string(v_row));
  oss << ", v_col=" << (v_col_dynamic ? "?" : std::to_string(v_col));
  oss << ", blayout=" << TileLayoutToStr(blayout);
  oss << ", slayout=" << TileLayoutToStr(slayout);
  oss << ", fractal=" << fractal;
  oss << ", pad=" << static_cast<int>(pad);
  if (compact != ir::CompactMode::null) {
    oss << ", compact=" << static_cast<int>(compact);
  }
  oss << ">";
  return oss.str();
}

// Extract dtype, shape and layout from a TileType into output parameters
static void ExtractTileTypeInfo(const TileType& tile_type, const PTOCodegen& codegen, std::string& dtype_str,
                                int64_t& rows, int64_t& cols, ir::TileLayout& blayout,
                                ir::TileLayout& slayout, uint64_t& fractal, ir::TilePad& pad,
                                int64_t& v_row, int64_t& v_col, bool& v_row_dynamic, bool& v_col_dynamic,
                                ir::CompactMode& compact) {
  dtype_str = codegen.GetTypeString(tile_type.dtype_);
  if (tile_type.shape_.size() >= 2) {
    if (auto c0 = As<ir::ConstInt>(tile_type.shape_[0])) rows = c0->value_;
    if (auto c1 = As<ir::ConstInt>(tile_type.shape_[1])) cols = c1->value_;
  } else if (tile_type.shape_.size() == 1) {
    if (auto c0 = As<ir::ConstInt>(tile_type.shape_[0])) {
      rows = 1;
      cols = c0->value_;
    }
  }
  v_row = rows;
  v_col = cols;
  if (tile_type.tile_view_.has_value()) {
    const auto& tv = *tile_type.tile_view_;
    blayout = tv.blayout;
    slayout = tv.slayout;
    fractal = tv.fractal;
    pad = tv.pad;
    compact = tv.compact;
    if (tv.valid_shape.size() >= 1) {
      if (auto var = As<ir::Var>(tv.valid_shape[0])) {
        v_row_dynamic = true;
      } else if (auto c = As<ir::ConstInt>(tv.valid_shape[0])) {
        if (c->value_ == -1) {
          v_row_dynamic = true;
        } else {
          v_row = c->value_;
        }
      }
    }
    if (tv.valid_shape.size() >= 2) {
      if (auto var = As<ir::Var>(tv.valid_shape[1])) {
        v_col_dynamic = true;
      } else if (auto c = As<ir::ConstInt>(tv.valid_shape[1])) {
        if (c->value_ == -1) {
          v_col_dynamic = true;
        } else {
          v_col = c->value_;
        }
      }
    }
  } else if (cols == 1 && rows > 1) {
    blayout = ir::TileLayout::col_major;
  }
}

std::string PTOCodegen::GetTileBufTypeString(const ir::MemRef* memref) const {
  std::string loc = MemorySpaceToMLIR(memref->memory_space_);
  std::string dtype_str = "f32";
  int64_t rows = 32;
  int64_t cols = 32;
  ir::TileLayout blayout = ir::TileLayout::row_major;
  ir::TileLayout slayout = ir::TileLayout::none_box;
  uint64_t fractal = 512;
  ir::TilePad pad = ir::TilePad::null;
  int64_t v_row = 32;
  int64_t v_col = 32;
  bool v_row_dynamic = false;
  bool v_col_dynamic = false;
  ir::CompactMode compact = ir::CompactMode::null;

  auto tile_it = memref_to_tile_type_.find(memref);
  if (tile_it != memref_to_tile_type_.end()) {
    ExtractTileTypeInfo(*tile_it->second, *this, dtype_str, rows, cols, blayout, slayout, fractal, pad,
                        v_row, v_col, v_row_dynamic, v_col_dynamic, compact);
  }

  return FormatTileBufTypeString(loc, dtype_str, rows, cols, blayout, slayout, fractal, pad, v_row, v_col,
                                 v_row_dynamic, v_col_dynamic, compact);
}

std::string PTOCodegen::GetTileBufTypeStringFromTileType(
    const std::shared_ptr<const ir::TileType>& tile_type) const {
  INTERNAL_CHECK(tile_type) << "Internal error: tile_type must not be null";
  INTERNAL_CHECK(tile_type->memref_.has_value()) << "Internal error: tile_type must have a memref";

  std::string loc = MemorySpaceToMLIR(tile_type->memref_.value()->memory_space_);
  std::string dtype_str = "f32";
  int64_t rows = 32;
  int64_t cols = 32;
  ir::TileLayout blayout = ir::TileLayout::row_major;
  ir::TileLayout slayout = ir::TileLayout::none_box;
  uint64_t fractal = 512;
  ir::TilePad pad = ir::TilePad::null;
  int64_t v_row = 32;
  int64_t v_col = 32;
  bool v_row_dynamic = false;
  bool v_col_dynamic = false;
  ir::CompactMode compact = ir::CompactMode::null;

  ExtractTileTypeInfo(*tile_type, *this, dtype_str, rows, cols, blayout, slayout, fractal, pad,
                      v_row, v_col, v_row_dynamic, v_col_dynamic, compact);

  return FormatTileBufTypeString(loc, dtype_str, rows, cols, blayout, slayout, fractal, pad, v_row, v_col,
                                 v_row_dynamic, v_col_dynamic, compact);
}

std::string PTOCodegen::GetTileBufTypeStringWithPadOverride(
    const std::shared_ptr<const ir::TileType>& tile_type, ir::TilePad pad) const {
  INTERNAL_CHECK(tile_type) << "Internal error: tile_type must not be null";
  INTERNAL_CHECK(tile_type->memref_.has_value()) << "Internal error: tile_type must have a memref";

  std::string loc = MemorySpaceToMLIR(tile_type->memref_.value()->memory_space_);
  std::string dtype_str = "f32";
  int64_t rows = 32;
  int64_t cols = 32;
  ir::TileLayout blayout = ir::TileLayout::row_major;
  ir::TileLayout slayout = ir::TileLayout::none_box;
  uint64_t fractal = 512;
  ir::TilePad original_pad = ir::TilePad::null;
  int64_t v_row = 32;
  int64_t v_col = 32;
  bool v_row_dynamic = false;
  bool v_col_dynamic = false;
  ir::CompactMode compact = ir::CompactMode::null;

  ExtractTileTypeInfo(*tile_type, *this, dtype_str, rows, cols, blayout, slayout, fractal, original_pad,
                      v_row, v_col, v_row_dynamic, v_col_dynamic, compact);

  return FormatTileBufTypeString(loc, dtype_str, rows, cols, blayout, slayout, fractal, pad, v_row, v_col,
                                 v_row_dynamic, v_col_dynamic, compact);
}

std::string PTOCodegen::GetExprTypeAnnotation(const ir::ExprPtr& expr) {
  // Handle TupleGetItemExpr by looking up the underlying tile element directly
  if (auto tgi = As<ir::TupleGetItemExpr>(expr)) {
    if (auto tuple_var = As<ir::Var>(tgi->tuple_)) {
      auto it = tuple_var_to_make_tuple_.find(tuple_var->name_);
      if (it != tuple_var_to_make_tuple_.end()) {
        const auto& elems = it->second->elements_;
        if (tgi->index_ >= 0 && tgi->index_ < static_cast<int>(elems.size())) {
          return GetExprTypeAnnotation(elems[tgi->index_]);
        }
      }
    }
  }
  if (auto var = As<ir::Var>(expr)) {
    // Check if variable was remapped to a dynamically-allocated tile buffer (e.g., reshape output)
    auto mlir_it = var_to_mlir_.find(var->name_);
    if (mlir_it != var_to_mlir_.end()) {
      auto extra_it = extra_tile_buf_types_.find(mlir_it->second);
      if (extra_it != extra_tile_buf_types_.end()) {
        return extra_it->second;
      }
    }
    // Check if this variable maps to a tile buffer via memref
    auto memref_it = var_to_memref_.find(var->name_);
    if (memref_it != var_to_memref_.end()) {
      return GetTileBufTypeString(memref_it->second);
    }
    // Check if this is a scalar parameter
    if (auto scalar_type = As<ScalarType>(var->GetType())) {
      return GetTypeString(scalar_type->dtype_);
    }
    // Check if variable has TileType with memref
    if (auto tile_type = As<TileType>(var->GetType())) {
      if (tile_type->memref_.has_value()) {
        return GetTileBufTypeString(tile_type->memref_.value().get());
      }
      return "";
    }
  }
  if (auto const_float = As<ir::ConstFloat>(expr)) {
    return "f32";
  }
  if (auto const_int = As<ir::ConstInt>(expr)) {
    return "index";
  }
  return "";
}

std::string PTOCodegen::GetCurrentResultTileBufTypeString() const {
  if (current_result_tile_type_ && current_result_tile_type_->memref_.has_value()) {
    return GetTileBufTypeStringFromTileType(current_result_tile_type_);
  }
  return "";
}

std::string PTOCodegen::GetCurrentResultVarName() const { return current_result_var_name_; }

void PTOCodegen::SetVarMlirName(const std::string& ir_name, const std::string& mlir_name) {
  var_to_mlir_[ir_name] = mlir_name;
}

void PTOCodegen::SetTensorViewName(const std::string& ir_name, const std::string& mlir_name) {
  var_to_mlir_[ir_name] = mlir_name;
  tensor_to_view_[ir_name] = mlir_name;
}

void PTOCodegen::UpdateTileValidShape(const std::string tile, const std::string& row,
                                      const std::string& col) {
  tile_valid_shapes_[tile] = {row, col};
}

std::pair<std::string, std::string> PTOCodegen::GetTileValidShape(const std::string tile) const {
  auto it = tile_valid_shapes_.find(tile);
  INTERNAL_CHECK(it != tile_valid_shapes_.end()) << "Tile valid_shape not found in mapping";
  return it->second;
}

bool PTOCodegen::IsDynamicTileType(const std::string tile_type) const {
  return tile_type.find("v_row=?, v_col=?") != std::string::npos;
}

bool PTOCodegen::IsDynamicTile(const std::shared_ptr<const ir::TileType>& tile_type) {
  if (!tile_type || !tile_type->tile_view_.has_value()) return false;
  const auto& vs = tile_type->tile_view_->valid_shape;
  for (const auto& dim : vs) {
    if (As<ir::Var>(dim)) return true;
    if (auto c = As<ir::ConstInt>(dim)) {
      if (c->value_ == -1) return true;
    }
  }
  return false;
}

// ========================================================================
// Control flow helpers
// ========================================================================

std::string PTOCodegen::EmitArithBinaryOp(const std::string& mlir_op, const std::string& lhs,
                                          const std::string& rhs, const std::string& result_type) {
  std::string result = NewTemp();
  Emit(result + " = " + mlir_op + " " + lhs + ", " + rhs + " : " + result_type);
  return result;
}

std::string PTOCodegen::EmitArithCmpi(const std::string& predicate, const std::string& lhs,
                                      const std::string& rhs, const std::string& operand_type) {
  std::string result = NewTemp();
  Emit(result + " = arith.cmpi " + predicate + ", " + lhs + ", " + rhs + " : " + operand_type);
  return result;
}

void PTOCodegen::VisitBinaryArithExpr(const BinaryExprPtr& op, const std::string& int_op,
                                      const std::string& float_op) {
  VisitExpr(op->left_);
  std::string lhs = current_expr_value_;
  VisitExpr(op->right_);
  std::string rhs = current_expr_value_;

  // Determine type: use float op for float types, int op otherwise
  std::string result_type = "index";
  std::string mlir_op = int_op;
  if (auto scalar_type = As<ScalarType>(op->GetType())) {
    if (scalar_type->dtype_.IsFloat()) {
      result_type = GetTypeString(scalar_type->dtype_);
      mlir_op = float_op;
    }
  }
  current_expr_value_ = EmitArithBinaryOp(mlir_op, lhs, rhs, result_type);
}

void PTOCodegen::VisitCmpExpr(const BinaryExprPtr& op, const std::string& predicate) {
  VisitExpr(op->left_);
  std::string lhs = current_expr_value_;
  VisitExpr(op->right_);
  std::string rhs = current_expr_value_;

  // Determine operand type from the left operand
  std::string operand_type = "index";
  bool is_float = false;
  if (auto scalar_type = As<ScalarType>(op->left_->GetType())) {
    if (scalar_type->dtype_.IsFloat()) {
      operand_type = GetTypeString(scalar_type->dtype_);
      is_float = true;
    }
  }

  if (is_float) {
    static const std::map<std::string, std::string> pred_map = {
        {"eq", "oeq"}, {"ne", "one"}, {"slt", "olt"}, {"sle", "ole"}, {"sgt", "ogt"}, {"sge", "oge"}};
    auto it = pred_map.find(predicate);
    INTERNAL_CHECK(it != pred_map.end()) << "Unsupported float predicate for " << predicate;
    std::string float_pred = it->second;

    std::string result = NewTemp();
    Emit(result + " = arith.cmpf " + float_pred + ", " + lhs + ", " + rhs + " : " + operand_type);
    current_expr_value_ = result;
  } else {
    current_expr_value_ = EmitArithCmpi(predicate, lhs, rhs, operand_type);
  }
}

// ========================================================================
// Expression visitors - Leaf nodes
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::VarPtr& op) { current_expr_value_ = GetVarName(op); }

void PTOCodegen::VisitExpr_(const ir::TupleGetItemExprPtr& op) {
  if (auto tuple_var = As<ir::Var>(op->tuple_)) {
    auto it = tuple_var_to_make_tuple_.find(tuple_var->name_);
    if (it != tuple_var_to_make_tuple_.end()) {
      const auto& elems = it->second->elements_;
      if (op->index_ >= 0 && op->index_ < static_cast<int>(elems.size())) {
        VisitExpr(elems[op->index_]);  // visits tile_a/tile_b directly
        return;
      }
    }
  }
  // Fallback: visit the tuple expression
  VisitExpr(op->tuple_);
}

void PTOCodegen::VisitExpr_(const ir::IterArgPtr& op) {
  current_expr_value_ = GetVarName(std::dynamic_pointer_cast<const ir::Var>(op));
}

void PTOCodegen::VisitExpr_(const ir::ConstIntPtr& op) {
  current_expr_value_ = GetOrEmitIndexConstant(op->value_);
}

void PTOCodegen::VisitExpr_(const ir::ConstFloatPtr& op) {
  std::string mlir_type = "f32";
  if (auto scalar_type = As<ScalarType>(op->GetType())) {
    mlir_type = GetTypeString(scalar_type->dtype_);
  }
  current_expr_value_ = GetOrEmitFloatConstant(op->value_, mlir_type);
}

void PTOCodegen::VisitExpr_(const ir::ConstBoolPtr& op) {
  std::string result = NewTemp();
  std::string val = op->value_ ? "1" : "0";
  Emit(result + " = arith.constant " + val + " : i1");
  current_expr_value_ = result;
}

// ========================================================================
// Expression visitors - Binary arithmetic
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::AddPtr& op) { VisitBinaryArithExpr(op, "arith.addi", "arith.addf"); }
void PTOCodegen::VisitExpr_(const ir::SubPtr& op) { VisitBinaryArithExpr(op, "arith.subi", "arith.subf"); }
void PTOCodegen::VisitExpr_(const ir::MulPtr& op) { VisitBinaryArithExpr(op, "arith.muli", "arith.mulf"); }
void PTOCodegen::VisitExpr_(const ir::FloorDivPtr& op) {
  VisitBinaryArithExpr(op, "arith.divsi", "arith.divf");
}
void PTOCodegen::VisitExpr_(const ir::FloorModPtr& op) {
  VisitBinaryArithExpr(op, "arith.remsi", "arith.remf");
}

// ========================================================================
// Expression visitors - Comparisons
// ========================================================================

void PTOCodegen::VisitExpr_(const ir::EqPtr& op) { VisitCmpExpr(op, "eq"); }
void PTOCodegen::VisitExpr_(const ir::NePtr& op) { VisitCmpExpr(op, "ne"); }
void PTOCodegen::VisitExpr_(const ir::LtPtr& op) { VisitCmpExpr(op, "slt"); }
void PTOCodegen::VisitExpr_(const ir::LePtr& op) { VisitCmpExpr(op, "sle"); }
void PTOCodegen::VisitExpr_(const ir::GtPtr& op) { VisitCmpExpr(op, "sgt"); }
void PTOCodegen::VisitExpr_(const ir::GePtr& op) { VisitCmpExpr(op, "sge"); }

void PTOCodegen::VisitExpr_(const ir::MaxPtr& op) { VisitBinaryArithExpr(op, "arith.maxsi", "arith.maximumf"); }
void PTOCodegen::VisitExpr_(const ir::MinPtr& op) { VisitBinaryArithExpr(op, "arith.minsi", "arith.minimumf"); }

void PTOCodegen::VisitExpr_(const ir::NotPtr& op) {
  VisitExpr(op->operand_);
  std::string operand = current_expr_value_;
  current_expr_value_ = "";

  // Logical NOT: arith.xori %val, %true : i1
  std::string true_val = NewTemp();
  Emit(true_val + " = arith.constant 1 : i1");
  std::string result = NewTemp();
  Emit(result + " = arith.xori " + operand + ", " + true_val + " : i1");
  current_expr_value_ = result;
}

// ========================================================================
// Statement visitors - Control flow
// ========================================================================

void PTOCodegen::VisitStmt_(const EvalStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null EvalStmt";
  INTERNAL_CHECK(op->expr_ != nullptr) << "Internal error: EvalStmt has null expression";
  VisitExpr(op->expr_);
}

void PTOCodegen::VisitStmt_(const YieldStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null YieldStmt";

  if (op->value_.empty()) {
    return;
  }

  std::vector<std::string> yielded_values;
  for (const auto& expr : op->value_) {
    VisitExpr(expr);
    std::string val = current_expr_value_;
    current_expr_value_ = "";

    if (indirect_select_depth_ > 0) {
      if (auto tile_type = As<TileType>(expr->GetType())) {
        // Yield i64 addr instead of tile_buf for TileType in indirect-select scf.if.
        // If expr is a Var whose name is in indirect_addr_vars_, it was already mapped
        // directly to an addr_ssa by the inner IfStmt reconstruction (to avoid emitting
        // a dead pto.alloc_tile). In that case, val already holds the correct i64 addr —
        // don't override with the static memref addr.
        bool is_indirect_addr_var = false;
        if (auto var = As<ir::Var>(expr)) {
          is_indirect_addr_var = (indirect_addr_vars_.count(var->name_) > 0);
        }
        if (!is_indirect_addr_var) {
          std::string addr_operand;
          if (tile_type->memref_.has_value() && tile_type->memref_.value()->addr_) {
            if (auto ca = As<ir::ConstInt>(tile_type->memref_.value()->addr_)) {
              addr_operand = GetOrEmitI64Constant(ca->value_);
            }
          }
          if (addr_operand.empty()) addr_operand = GetOrEmitI64Constant(0);
          val = addr_operand;
        }
        // else: val already holds addr_ssa (e.g., "%57") — keep as-is
      } else if (auto tensor_type = As<TensorType>(expr->GetType())) {
        // TensorType: yield index offset instead of ptr for indirect-select scf.if.
        // The scf.if yields an index (addptr offset), then IfStmt reconstruction emits
        // pto.addptr(base, selected_offset) + pto.make_tensor_view to rebuild the view.
        INTERNAL_CHECK(tensor_type->tensor_view_.has_value() &&
                       tensor_type->tensor_view_->ptr.has_value())
            << "TensorType yield: no ptr in TensorView. Ensure the workspace tensor "
            << "variable has no type annotation (write `ws = pl.make_tensor(...)`, "
            << "not `ws: pl.Tensor[...] = pl.make_tensor(...)`)";
        auto& ptr_expr = *tensor_type->tensor_view_->ptr;
        auto ptr_ty = As<PtrType>(ptr_expr->GetType());
        INTERNAL_CHECK(ptr_ty && ptr_ty->offset.has_value())
            << "TensorType yield: ptr has no offset. Use pl.addptr(workspace, offset).";
        val = GetExprAsCode(*ptr_ty->offset);
      }
    }
    yielded_values.push_back(val);
  }

  yield_buffer_ = yielded_values;
}

void PTOCodegen::VisitStmt_(const ir::SectionStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null SectionStmt";
  INTERNAL_CHECK(op->body_ != nullptr) << "Internal error: SectionStmt has null body";

  // Determine the section name based on section_kind
  std::string section_name;
  switch (op->section_kind_) {
    case ir::SectionKind::Vector:
      section_name = "vector";
      break;
    case ir::SectionKind::Cube:
      section_name = "cube";
      break;
    default:
      throw pypto::ValueError("Unknown SectionKind in SectionStmt");
  }

  // Emit pto.section.{vector|cube} {
  Emit("pto.section." + section_name + " {");
  indent_level_++;

  // Emit section-specific tile allocations at the top of the section
  const std::set<const ir::MemRef*>* section_memrefs = nullptr;
  if (op->section_kind_ == ir::SectionKind::Cube && !cube_only_memrefs_.empty())
    section_memrefs = &cube_only_memrefs_;
  if (op->section_kind_ == ir::SectionKind::Vector && !vec_only_memrefs_.empty())
    section_memrefs = &vec_only_memrefs_;
  if (section_memrefs)
    EmitAllocTiles(nullptr, all_memrefs_, section_memrefs);

  VisitStmt(op->body_);
  indent_level_--;
  Emit("}");
}

void PTOCodegen::VisitStmt_(const IfStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null IfStmt";
  INTERNAL_CHECK(op->condition_ != nullptr) << "Internal error: IfStmt has null condition";
  INTERNAL_CHECK(op->then_body_ != nullptr) << "Internal error: IfStmt has null then_body";

  // Evaluate condition
  VisitExpr(op->condition_);
  std::string condition = current_expr_value_;
  current_expr_value_ = "";

  if (op->return_vars_.empty()) {
    // Simple scf.if (no return values)
    auto if_entry_var_to_mlir = var_to_mlir_;
    Emit("scf.if " + condition + " {");
    indent_level_++;
    var_to_mlir_ = if_entry_var_to_mlir;
    VisitStmt(op->then_body_);
    var_to_mlir_ = if_entry_var_to_mlir;
    indent_level_--;

    if (op->else_body_.has_value()) {
      Emit("} else {");
      indent_level_++;
      var_to_mlir_ = if_entry_var_to_mlir;
      VisitStmt(*op->else_body_);
      var_to_mlir_ = if_entry_var_to_mlir;
      indent_level_--;
    }
    var_to_mlir_ = if_entry_var_to_mlir;
    Emit("}");
  } else {
    // scf.if with return values
    std::vector<std::string> return_var_names;
    std::vector<std::string> return_var_types;
    std::vector<bool> needs_indirect_yield;  // per-return-var flag

    for (const auto& return_var : op->return_vars_) {
      std::string ret_name = NewTemp();
      var_to_mlir_[return_var->name_] = ret_name;
      return_var_names.push_back(ret_name);
      // Default to index for scalar types
      std::string type_str = "index";
      bool indirect_yield = false;
      if (auto scalar_type = As<ScalarType>(return_var->GetType())) {
        if (scalar_type->dtype_ == DataType::BOOL) {
          type_str = "i1";
        } else if (scalar_type->dtype_.IsFloat()) {
          type_str = GetTypeString(scalar_type->dtype_);
        }
      } else if (As<TileType>(return_var->GetType())) {
        // TileType: scf.if yields i64 addr; pto.alloc_tile reconstructs tile_buf after scf.if.
        type_str = "i64";
        indirect_yield = true;
      } else if (auto tensor_type = As<TensorType>(return_var->GetType())) {
        // TensorType: scf.if yields index offset; pto.addptr + pto.make_tensor_view reconstruct after.
        type_str = "index";
        indirect_yield = true;
      }
      return_var_types.push_back(type_str);
      needs_indirect_yield.push_back(indirect_yield);
    }

    bool any_indirect_yield = std::any_of(needs_indirect_yield.begin(), needs_indirect_yield.end(),
                                          [](bool b) { return b; });

    CHECK(op->else_body_.has_value()) << "IfStmt with return_vars requires else_body";
    auto if_entry_var_to_mlir = var_to_mlir_;

    // Emit: %ret0, %ret1 = scf.if %cond -> (type0, type1) {
    std::ostringstream oss;
    for (size_t i = 0; i < return_var_names.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << return_var_names[i];
    }
    oss << " = scf.if " << condition << " -> (";
    for (size_t i = 0; i < return_var_types.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << return_var_types[i];
    }
    oss << ") {";
    Emit(oss.str());
    indent_level_++;

    // Enter indirect-select mode if any return var needs indirect yield
    if (any_indirect_yield) indirect_select_depth_++;

    // Then branch
    yield_buffer_.clear();
    var_to_mlir_ = if_entry_var_to_mlir;
    VisitStmt(op->then_body_);
    var_to_mlir_ = if_entry_var_to_mlir;
    if (!yield_buffer_.empty()) {
      std::ostringstream yield_oss;
      yield_oss << "scf.yield ";
      for (size_t i = 0; i < yield_buffer_.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << yield_buffer_[i];
      }
      yield_oss << " : ";
      for (size_t i = 0; i < return_var_types.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << return_var_types[i];
      }
      Emit(yield_oss.str());
    }
    CHECK(yield_buffer_.size() == return_var_types.size())
        << "IfStmt then-branch yield count (" << yield_buffer_.size() << ") must match return_vars ("
        << return_var_types.size() << ")";
    yield_buffer_.clear();
    indent_level_--;

    // Else branch
    if (op->else_body_.has_value()) {
      Emit("} else {");
      indent_level_++;
      var_to_mlir_ = if_entry_var_to_mlir;
      VisitStmt(*op->else_body_);
      var_to_mlir_ = if_entry_var_to_mlir;
      if (!yield_buffer_.empty()) {
        std::ostringstream yield_oss;
        yield_oss << "scf.yield ";
        for (size_t i = 0; i < yield_buffer_.size(); ++i) {
          if (i > 0) yield_oss << ", ";
          yield_oss << yield_buffer_[i];
        }
        yield_oss << " : ";
        for (size_t i = 0; i < return_var_types.size(); ++i) {
          if (i > 0) yield_oss << ", ";
          yield_oss << return_var_types[i];
        }
        Emit(yield_oss.str());
      }
      CHECK(yield_buffer_.size() == return_var_types.size())
          << "IfStmt else-branch yield count (" << yield_buffer_.size() << ") must match return_vars ("
          << return_var_types.size() << ")";
      yield_buffer_.clear();
      indent_level_--;
    }
    var_to_mlir_ = if_entry_var_to_mlir;
    Emit("}");

    // Exit indirect-select mode
    if (any_indirect_yield) indirect_select_depth_--;

    // For each indirect-yield return var, reconstruct the real object from the scalar returned by scf.if
    for (size_t rv_idx = 0; rv_idx < op->return_vars_.size(); ++rv_idx) {
      if (!needs_indirect_yield[rv_idx]) continue;
      const auto& return_var = op->return_vars_[rv_idx];
      if (auto tile_type = As<TileType>(return_var->GetType())) {
        std::string addr_ssa = return_var_names[rv_idx];
        if (indirect_select_depth_ > 0) {
          // Inside an outer indirect-select chain: skip pto.alloc_tile.
          // Map var directly to addr_ssa so the outer yield can propagate it as i64
          // without overriding with the static memref addr.
          var_to_mlir_[return_var->name_] = addr_ssa;
          indirect_addr_vars_.insert(return_var->name_);
        } else {
          // Outermost level: reconstruct tile_buf from the dynamically selected addr.
          std::string tile_buf_type = GetTileBufTypeStringFromTileType(tile_type);
          std::string tile_name = NewTemp();

          std::string valid_row_mlir;
          std::string valid_col_mlir;
          bool is_dynamic = IsDynamicTileType(tile_buf_type);
          if (is_dynamic) {
            valid_row_mlir = GetExprAsCode(tile_type->shape_[0]);
            valid_col_mlir = GetExprAsCode(tile_type->shape_[1]);
          }

          std::ostringstream alloc_line;
          alloc_line << tile_name << " = pto.alloc_tile addr = " << addr_ssa;
          if (!valid_row_mlir.empty()) alloc_line << " valid_row = " << valid_row_mlir;
          if (!valid_col_mlir.empty()) alloc_line << " valid_col = " << valid_col_mlir;
          alloc_line << " : " << tile_buf_type;
          Emit(alloc_line.str());
          var_to_mlir_[return_var->name_] = tile_name;
          extra_tile_buf_types_[tile_name] = tile_buf_type;
          indirect_addr_vars_.erase(return_var->name_);

          if (is_dynamic) {
            UpdateTileValidShape(tile_name, valid_row_mlir, valid_col_mlir);
          }
        }
      } else if (auto tensor_type = As<TensorType>(return_var->GetType())) {
        // TensorType: reconstruct tensor_view from (base ptr, selected index offset) returned by scf.if.
        // 1. Read base_ptr from the PtrType annotation stored in the tensor's TensorView::ptr.
        INTERNAL_CHECK(tensor_type->tensor_view_.has_value() && tensor_type->tensor_view_->ptr.has_value())
            << "TensorType indirect-select: no ptr in TensorView for " << return_var->name_
            << ". Ensure the workspace tensor has no type annotation (use `ws = pl.make_tensor(...)`)";
        auto& ptr_expr_ref = *tensor_type->tensor_view_->ptr;
        auto ptr_ty = As<PtrType>(ptr_expr_ref->GetType());
        INTERNAL_CHECK(ptr_ty && ptr_ty->base_ptr.has_value())
            << "TensorType indirect-select: no base_ptr in PtrType for " << return_var->name_
            << ". Use pl.addptr(workspace, offset).";
        std::string base_ssa = GetExprAsCode(*ptr_ty->base_ptr);
        // 2. Emit: cur_ptr = pto.addptr base, selected_offset : type -> type
        std::string dtype_str = GetTypeString(tensor_type->dtype_);
        std::string ptr_type = "!pto.ptr<" + dtype_str + ">";
        std::string cur_ptr = NewTemp();
        Emit(cur_ptr + " = pto.addptr " + base_ssa + ", " + return_var_names[rv_idx] +
             " : " + ptr_type + " -> " + ptr_type);
        // 3. Emit: view = pto.make_tensor_view cur_ptr, shape, strides
        std::string view_name = NewTemp();
        std::ostringstream oss;
        oss << view_name << " = pto.make_tensor_view " << cur_ptr << ", shape = [";
        for (size_t j = 0; j < tensor_type->shape_.size(); j++) {
          if (j > 0) oss << ", ";
          oss << GetExprAsCode(tensor_type->shape_[j]);
        }
        oss << "], strides = [";
        const bool has_stride = tensor_type->tensor_view_.has_value() &&
                                !tensor_type->tensor_view_->stride.empty();
        if (has_stride) {
          const auto& stride = tensor_type->tensor_view_->stride;
          for (size_t j = 0; j < stride.size(); j++) {
            if (j > 0) oss << ", ";
            oss << GetExprAsCode(stride[j]);
          }
        } else {
          if (tensor_type->shape_.size() == 2) {
            oss << GetExprAsCode(tensor_type->shape_[1]) << ", " << GetOrEmitIndexConstant(1);
          } else if (tensor_type->shape_.size() == 1) {
            oss << GetOrEmitIndexConstant(1);
          }
        }
        oss << "] : " << GetTensorViewTypeString(tensor_type.get());
        Emit(oss.str());
        SetTensorViewName(return_var->name_, view_name);
      } else {
        INTERNAL_CHECK(false) << "Internal error: unsupported type for indirect-select return var: "
                              << return_var->name_;
      }
    }
  }
}

void PTOCodegen::VisitStmt_(const ForStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ForStmt";
  INTERNAL_CHECK(op->loop_var_ != nullptr) << "Internal error: ForStmt has null loop_var";
  INTERNAL_CHECK(op->body_ != nullptr) << "Internal error: ForStmt has null body";

  CHECK(op->iter_args_.size() == op->return_vars_.size())
      << "ForStmt iter_args size (" << op->iter_args_.size() << ") must equal return_vars size ("
      << op->return_vars_.size() << ")";

  if (op->kind_ == ir::ForKind::Unroll) {
    LOG_WARN << "ForKind::Unroll loop was not expanded before codegen; "
                "generating sequential loop as fallback";
  }

  // Evaluate loop bounds
  VisitExpr(op->start_);
  std::string start = current_expr_value_;
  current_expr_value_ = "";

  VisitExpr(op->stop_);
  std::string stop = current_expr_value_;
  current_expr_value_ = "";

  VisitExpr(op->step_);
  std::string step = current_expr_value_;
  current_expr_value_ = "";

  // Register loop variable
  std::string loop_var_name = NewTemp();
  var_to_mlir_[op->loop_var_->name_] = loop_var_name;

  if (op->iter_args_.empty()) {
    // Simple scf.for (no iter_args)
    Emit("scf.for " + loop_var_name + " = " + start + " to " + stop + " step " + step + " {");
    indent_level_++;

    yield_buffer_.clear();
    VisitStmt(op->body_);

    indent_level_--;
    Emit("}");
  } else {
    // scf.for with iter_args
    std::vector<std::string> init_values;
    std::vector<std::string> iter_arg_names;
    std::vector<std::string> iter_arg_types;

    for (const auto& iter_arg : op->iter_args_) {
      VisitExpr(iter_arg->initValue_);
      init_values.push_back(current_expr_value_);
      current_expr_value_ = "";

      std::string iter_name = NewTemp();
      var_to_mlir_[iter_arg->name_] = iter_name;
      iter_arg_names.push_back(iter_name);

      std::string type_str = "index";
      if (auto scalar_type = As<ScalarType>(iter_arg->GetType())) {
        if (scalar_type->dtype_ == DataType::BOOL) {
          type_str = "i1";
        } else if (scalar_type->dtype_.IsFloat()) {
          type_str = GetTypeString(scalar_type->dtype_);
        }
      }
      iter_arg_types.push_back(type_str);
    }

    // Register return_vars SSA names
    std::vector<std::string> return_var_names;
    for (const auto& return_var : op->return_vars_) {
      std::string ret_name = NewTemp();
      var_to_mlir_[return_var->name_] = ret_name;
      return_var_names.push_back(ret_name);
    }

    // Emit: %ret0 = scf.for %i = %start to %stop step %step
    //           iter_args(%acc = %init) -> (type) {
    std::ostringstream oss;
    for (size_t i = 0; i < return_var_names.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << return_var_names[i];
    }
    oss << " = scf.for " << loop_var_name << " = " << start << " to " << stop << " step " << step;
    oss << " iter_args(";
    for (size_t i = 0; i < iter_arg_names.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << iter_arg_names[i] << " = " << init_values[i];
    }
    oss << ") -> (";
    for (size_t i = 0; i < iter_arg_types.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << iter_arg_types[i];
    }
    oss << ") {";
    Emit(oss.str());
    indent_level_++;

    yield_buffer_.clear();
    VisitStmt(op->body_);

    // Emit scf.yield from yield_buffer_
    CHECK(!yield_buffer_.empty())
        << "ForStmt with iter_args must have a pl.yield_() statement in the body";
    std::ostringstream yield_oss;
    yield_oss << "scf.yield ";
    for (size_t i = 0; i < yield_buffer_.size(); ++i) {
      if (i > 0) yield_oss << ", ";
      yield_oss << yield_buffer_[i];
    }
    yield_oss << " : ";
    for (size_t i = 0; i < iter_arg_types.size(); ++i) {
      if (i > 0) yield_oss << ", ";
      yield_oss << iter_arg_types[i];
    }
    Emit(yield_oss.str());
    yield_buffer_.clear();

    indent_level_--;
    Emit("}");
  }
}

void PTOCodegen::VisitStmt_(const ir::WhileStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null WhileStmt";
  INTERNAL_CHECK(op->condition_ != nullptr) << "Internal error: WhileStmt has null condition";
  INTERNAL_CHECK(op->body_ != nullptr) << "Internal error: WhileStmt has null body";

  CHECK(op->iter_args_.size() == op->return_vars_.size())
      << "WhileStmt iter_args size (" << op->iter_args_.size() << ") must equal return_vars size ("
      << op->return_vars_.size() << ")";

  if (op->iter_args_.empty()) {
    // Standard MLIR scf.while with no iter_args:
    //   scf.while : () -> () {
    //     %cond = ...
    //     scf.condition(%cond)
    //   } do {
    //   ^bb0:
    //     ...body...
    //     scf.yield
    //   }
    // Note: "before" region has no ^bb0 header (no block args).
    //       "do" region has explicit ^bb0 even with no args.
    Emit("scf.while : () -> () {");
    indent_level_++;
    VisitExpr(op->condition_);
    std::string condition = current_expr_value_;
    current_expr_value_ = "";
    Emit("scf.condition(" + condition + ")");
    indent_level_--;
    Emit("} do {");
    indent_level_++;
    Emit("^bb0:");
    VisitStmt(op->body_);
    Emit("scf.yield");
    indent_level_--;
    Emit("}");
  } else {
    // Standard MLIR scf.while with iter_args:
    //   %ret0, %ret1 = scf.while (%arg0 = %init0, %arg1 = %init1)
    //       : (type0, type1) -> (type0, type1) {
    //     %cond = ...                               ← NO ^bb0 header; args are implicit
    //     scf.condition(%cond) %arg0, %arg1 : type0, type1
    //   } do {
    //   ^bb0(%arg0: type0, %arg1: type1):           ← explicit ^bb0 required
    //     ...body...
    //     scf.yield %new0, %new1 : type0, type1
    //   }
    std::vector<std::string> init_values;
    std::vector<std::string> iter_arg_names;
    std::vector<std::string> iter_arg_types;

    for (const auto& iter_arg : op->iter_args_) {
      VisitExpr(iter_arg->initValue_);
      init_values.push_back(current_expr_value_);
      current_expr_value_ = "";

      std::string iter_name = NewTemp();
      var_to_mlir_[iter_arg->name_] = iter_name;
      iter_arg_names.push_back(iter_name);

      std::string type_str = "index";
      if (auto scalar_type = As<ScalarType>(iter_arg->GetType())) {
        if (scalar_type->dtype_ == DataType::BOOL) {
          type_str = "i1";
        } else if (scalar_type->dtype_.IsFloat()) {
          type_str = GetTypeString(scalar_type->dtype_);
        }
      }
      iter_arg_types.push_back(type_str);
    }

    // Register return_vars SSA names
    std::vector<std::string> return_var_names;
    for (const auto& return_var : op->return_vars_) {
      std::string ret_name = NewTemp();
      var_to_mlir_[return_var->name_] = ret_name;
      return_var_names.push_back(ret_name);
    }

    // Build type list string (shared by header and both regions)
    std::ostringstream type_oss;
    for (size_t i = 0; i < iter_arg_types.size(); ++i) {
      if (i > 0) type_oss << ", ";
      type_oss << iter_arg_types[i];
    }
    std::string type_list = type_oss.str();

    // Build ^bb0(arg: type, ...) header string (shared by both regions)
    std::ostringstream bb0_oss;
    bb0_oss << "^bb0(";
    for (size_t i = 0; i < iter_arg_names.size(); ++i) {
      if (i > 0) bb0_oss << ", ";
      bb0_oss << iter_arg_names[i] << ": " << iter_arg_types[i];
    }
    bb0_oss << "):";
    std::string bb0_header = bb0_oss.str();

    // Emit: %ret0, %ret1 = scf.while (%arg0 = %init0) : (type0) -> (type0) {
    std::ostringstream header_oss;
    for (size_t i = 0; i < return_var_names.size(); ++i) {
      if (i > 0) header_oss << ", ";
      header_oss << return_var_names[i];
    }
    header_oss << " = scf.while (";
    for (size_t i = 0; i < iter_arg_names.size(); ++i) {
      if (i > 0) header_oss << ", ";
      header_oss << iter_arg_names[i] << " = " << init_values[i];
    }
    header_oss << ") : (" << type_list << ") -> (" << type_list << ") {";
    Emit(header_oss.str());
    indent_level_++;

    // "before" region: evaluate condition then scf.condition — no ^bb0 header, iter_args implicit
    VisitExpr(op->condition_);
    std::string condition = current_expr_value_;
    current_expr_value_ = "";
    std::ostringstream cond_oss;
    cond_oss << "scf.condition(" << condition << ") ";
    for (size_t i = 0; i < iter_arg_names.size(); ++i) {
      if (i > 0) cond_oss << ", ";
      cond_oss << iter_arg_names[i];
    }
    cond_oss << " : " << type_list;
    Emit(cond_oss.str());
    indent_level_--;

    // "do" region: body, then scf.yield new_values : types
    Emit("} do {");
    indent_level_++;
    Emit(bb0_header);

    yield_buffer_.clear();
    VisitStmt(op->body_);

    CHECK(!yield_buffer_.empty())
        << "WhileStmt with iter_args must have a pl.yield_() statement in the body";
    std::ostringstream yield_oss;
    yield_oss << "scf.yield ";
    for (size_t i = 0; i < yield_buffer_.size(); ++i) {
      if (i > 0) yield_oss << ", ";
      yield_oss << yield_buffer_[i];
    }
    yield_oss << " : " << type_list;
    Emit(yield_oss.str());
    yield_buffer_.clear();

    indent_level_--;
    Emit("}");
  }
}

void PTOCodegen::VisitStmt_(const ir::SeqStmtsPtr& op) {
  for (const auto& stmt : op->stmts_) {
    VisitStmt(stmt);
  }
}

void PTOCodegen::VisitStmt_(const ir::OpStmtsPtr& op) {
  for (const auto& stmt : op->stmts_) {
    VisitStmt(stmt);
  }
}

void PTOCodegen::VisitStmt_(const ir::BreakStmtPtr& op) {
  INTERNAL_CHECK(false) << "Internal error: BreakStmt reached PTOCodegen. "
                        << "LowerBreakContinue pass should have eliminated all BreakStmts before codegen.";
}

void PTOCodegen::VisitStmt_(const ir::ContinueStmtPtr& op) {
  INTERNAL_CHECK(false) << "Internal error: ContinueStmt reached PTOCodegen. "
                        << "LowerBreakContinue pass should have eliminated all ContinueStmts before codegen.";
}

// ========================================================================
// Helper function generation (FunctionType::Helper)
// ========================================================================

void PTOCodegen::GenerateHelperFunction(const FunctionPtr& func) {
  current_function_ = func;
  temp_counter_ = 0;
  last_assigned_temp_ = "";
  var_to_mlir_.clear();
  tensor_to_view_.clear();
  memref_to_mlir_.clear();
  var_to_memref_.clear();
  memref_to_tile_type_.clear();
  emitted_constants_.clear();
  emitted_i64_constants_.clear();
  tile_valid_shapes_.clear();
  constants_section_.str("");
  constants_section_.clear();
  current_expr_value_ = "";

  // Pre-register tile param memrefs so body ops can resolve them via GetExprTypeAnnotation
  for (size_t i = 0; i < func->params_.size(); ++i) {
    const auto& param = func->params_[i];
    std::string arg_name = "%arg" + std::to_string(i);
    if (auto tile_type = As<TileType>(param->GetType())) {
      if (tile_type->memref_.has_value()) {
        const ir::MemRef* memref = tile_type->memref_.value().get();
        memref_to_mlir_[memref] = arg_name;
        var_to_memref_[param->name_] = memref;
        memref_to_tile_type_[memref] = tile_type;
      } else {
        // TileType without memref (from DSL annotation): build type string directly
        std::string dtype_str = "f32";
        int64_t rows = 32, cols = 32;
        ir::TileLayout blayout = ir::TileLayout::row_major;
        ir::TileLayout slayout = ir::TileLayout::none_box;
        uint64_t fractal = 512;
        ir::TilePad pad = ir::TilePad::null;
        int64_t v_row = 32, v_col = 32;
        bool v_row_dynamic = false, v_col_dynamic = false;
        ir::CompactMode compact = ir::CompactMode::null;
        ExtractTileTypeInfo(*tile_type, *this, dtype_str, rows, cols, blayout, slayout, fractal, pad,
                            v_row, v_col, v_row_dynamic, v_col_dynamic, compact);
        std::string loc = "vec";  // default memory space for annotation-only tile params
        var_to_mlir_[param->name_] = arg_name;
        extra_tile_buf_types_[arg_name] = FormatTileBufTypeString(
            loc, dtype_str, rows, cols, blayout, slayout, fractal, pad, v_row, v_col,
            v_row_dynamic, v_col_dynamic, compact);
      }
    } else if (auto tensor_type = As<TensorType>(param->GetType())) {
      std::string view_name = NewTemp();
      tensor_to_view_[param->name_] = view_name;
    }
  }

  // Emit: func.func @name(%arg0: type, ...) -> ret_type {
  stream_ << GetIndent() << "func.func @" << func->name_ << "(";
  for (size_t i = 0; i < func->params_.size(); ++i) {
    if (i > 0) stream_ << ", ";
    const auto& param = func->params_[i];
    std::string arg_name = "%arg" + std::to_string(i);
    var_to_mlir_[param->name_] = arg_name;
    if (auto tensor_type = As<TensorType>(param->GetType())) {
      stream_ << arg_name << ": !pto.ptr<" << GetTypeString(tensor_type->dtype_) << ">";
    } else if (auto tile_type = As<TileType>(param->GetType())) {
      // Generate tile_buf type string; use memref's memory space if available,
      // otherwise default to "vec" (the most common AICore vector memory space).
      std::string dtype_str = "f32";
      int64_t rows = 32, cols = 32;
      ir::TileLayout blayout = ir::TileLayout::row_major;
      ir::TileLayout slayout = ir::TileLayout::none_box;
      uint64_t fractal = 512;
      ir::TilePad pad = ir::TilePad::null;
      int64_t v_row = 32, v_col = 32;
      bool v_row_dynamic = false, v_col_dynamic = false;
      ir::CompactMode compact = ir::CompactMode::null;
      ExtractTileTypeInfo(*tile_type, *this, dtype_str, rows, cols, blayout, slayout, fractal, pad,
                          v_row, v_col, v_row_dynamic, v_col_dynamic, compact);
      std::string loc = tile_type->memref_.has_value()
                            ? MemorySpaceToMLIR(tile_type->memref_.value()->memory_space_)
                            : "vec";
      stream_ << arg_name << ": "
              << FormatTileBufTypeString(loc, dtype_str, rows, cols, blayout, slayout, fractal, pad,
                                        v_row, v_col, v_row_dynamic, v_col_dynamic, compact);
    } else if (auto ptr_type = As<PtrType>(param->GetType())) {
      stream_ << arg_name << ": !pto.ptr<" << GetTypeString(ptr_type->dtype_) << ">";
    } else if (auto scalar_type = As<ScalarType>(param->GetType())) {
      stream_ << arg_name << ": " << GetTypeString(scalar_type->dtype_);
    } else {
      stream_ << arg_name << ": index";
    }
  }
  stream_ << ")";

  if (!func->return_types_.empty()) {
    stream_ << " -> ";
    const auto& rt = func->return_types_[0];
    if (auto scalar_type = As<ScalarType>(rt)) {
      stream_ << GetTypeString(scalar_type->dtype_);
    } else if (auto tensor_type = As<TensorType>(rt)) {
      stream_ << GetTensorViewTypeString(tensor_type.get());
    } else if (auto tile_type = As<TileType>(rt)) {
      std::string dtype_str = "f32";
      int64_t rows = 32, cols = 32;
      ir::TileLayout blayout = ir::TileLayout::row_major;
      ir::TileLayout slayout = ir::TileLayout::none_box;
      uint64_t fractal = 512;
      ir::TilePad pad = ir::TilePad::null;
      int64_t v_row = 32, v_col = 32;
      bool v_row_dynamic = false, v_col_dynamic = false;
      ir::CompactMode compact = ir::CompactMode::null;
      ExtractTileTypeInfo(*tile_type, *this, dtype_str, rows, cols, blayout, slayout, fractal, pad,
                          v_row, v_col, v_row_dynamic, v_col_dynamic, compact);
      std::string loc = tile_type->memref_.has_value()
                            ? MemorySpaceToMLIR(tile_type->memref_.value()->memory_space_)
                            : "vec";
      stream_ << FormatTileBufTypeString(loc, dtype_str, rows, cols, blayout, slayout, fractal, pad,
                                        v_row, v_col, v_row_dynamic, v_col_dynamic, compact);
    } else if (auto ptr_type = As<PtrType>(rt)) {
      stream_ << "!pto.ptr<" << GetTypeString(ptr_type->dtype_) << ">";
    } else {
      stream_ << "index";
    }
  }
  stream_ << " {\n";
  indent_level_++;
  function_body_indent_level_ = indent_level_;

  // Emit make_tensor_view preamble for any TensorType params (reuses existing method)
  EmitMakeTensorViews(func);

  if (func->body_) {
    // Capture body output in a temporary stream so we can emit constants first
    std::ostringstream saved_stream;
    saved_stream.swap(stream_);
    VisitStmt(func->body_);
    std::string body_content = stream_.str();
    stream_.swap(saved_stream);
    // Emit constants before body (constants may have been emitted during body traversal)
    if (!constants_section_.str().empty()) {
      stream_ << constants_section_.str();
      constants_section_.str("");
      constants_section_.clear();
    }
    stream_ << body_content;
  }

  // Add trailing func.return for void helpers that have no explicit return statement
  if (func->return_types_.empty()) {
    bool has_explicit_return = false;
    if (auto seq = As<ir::SeqStmts>(func->body_)) {
      if (!seq->stmts_.empty()) {
        has_explicit_return = As<ir::ReturnStmt>(seq->stmts_.back()) != nullptr;
      }
    }
    if (!has_explicit_return) {
      stream_ << GetIndent() << "return\n";
    }
  }

  indent_level_--;
  stream_ << GetIndent() << "}\n";
  function_body_indent_level_ = 0;
}

void PTOCodegen::EmitFuncCall(const CallPtr& op) {
  std::vector<std::string> arg_codes;
  std::vector<std::string> arg_type_strs;

  for (const auto& arg : op->args_) {
    arg_codes.push_back(GetExprAsCode(arg));
    std::string type_str;
    if (auto t = arg->GetType()) {
      if (auto scalar_type = As<ScalarType>(t)) {
        type_str = GetTypeString(scalar_type->dtype_);
      } else if (auto tensor_type = As<TensorType>(t)) {
        type_str = "!pto.ptr<" + GetTypeString(tensor_type->dtype_) + ">";
      } else if (auto tile_type = As<TileType>(t)) {
        type_str = GetExprTypeAnnotation(arg);
        if (type_str.empty()) type_str = "index";
      } else if (auto ptr_type = As<PtrType>(t)) {
        type_str = "!pto.ptr<" + GetTypeString(ptr_type->dtype_) + ">";
      } else {
        type_str = "index";
      }
    } else {
      type_str = "index";
    }
    arg_type_strs.push_back(type_str);
  }

  auto call_type = op->GetType();
  bool is_void = !call_type
      || (!As<ScalarType>(call_type) && !As<TensorType>(call_type)
          && !As<TileType>(call_type) && !As<PtrType>(call_type));
  std::ostringstream oss;
  if (is_void) {
    // void call: no result variable, no return type
    oss << "func.call @" << op->op_->name_ << "(";
    for (size_t i = 0; i < arg_codes.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << arg_codes[i];
    }
    oss << ") : (";
    for (size_t i = 0; i < arg_type_strs.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << arg_type_strs[i];
    }
    oss << ") -> ()";
    Emit(oss.str());
    current_expr_value_ = "";
  } else {
    std::string ret_type = "index";  // fallback for unrecognized types
    if (auto scalar_type = As<ScalarType>(call_type)) {
      ret_type = GetTypeString(scalar_type->dtype_);
    } else if (auto tensor_type = As<TensorType>(call_type)) {
      ret_type = GetTensorViewTypeString(tensor_type.get());
    } else if (auto tile_type = As<TileType>(call_type)) {
      ret_type = GetTileBufTypeStringFromTileType(tile_type);
    } else if (auto ptr_type = As<PtrType>(call_type)) {
      ret_type = "!pto.ptr<" + GetTypeString(ptr_type->dtype_) + ">";
    }
    std::string result = NewTemp();
    oss << result << " = func.call @" << op->op_->name_ << "(";
    for (size_t i = 0; i < arg_codes.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << arg_codes[i];
    }
    oss << ") : (";
    for (size_t i = 0; i < arg_type_strs.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << arg_type_strs[i];
    }
    oss << ") -> " << ret_type;
    Emit(oss.str());
    current_expr_value_ = result;
  }
}

void PTOCodegen::VisitStmt_(const ir::ReturnStmtPtr& op) {
  // For kernel functions (non-Helper), the trailing 'return' is emitted by
  // GenerateFunction; we skip the ReturnStmt visitor to avoid double emission.
  if (!current_function_ || current_function_->func_type_ != ir::FunctionType::Helper) {
    return;
  }
  if (op->value_.empty()) {
    Emit("return");
    return;
  }
  std::string val = GetExprAsCode(op->value_[0]);
  // Try GetExprTypeAnnotation first (handles tiles/tensors via lookup maps),
  // then fall back to type-based dispatch for scalars.
  std::string type_str = GetExprTypeAnnotation(op->value_[0]);
  if (type_str.empty()) {
    if (auto t = op->value_[0]->GetType()) {
      if (auto scalar_type = As<ScalarType>(t)) {
        type_str = GetTypeString(scalar_type->dtype_);
      } else if (auto tensor_type = As<TensorType>(t)) {
        type_str = GetTensorViewTypeString(tensor_type.get());
      } else if (auto ptr_type = As<PtrType>(t)) {
        type_str = "!pto.ptr<" + GetTypeString(ptr_type->dtype_) + ">";
      } else {
        type_str = "index";
      }
    } else {
      type_str = "index";
    }
  }
  Emit("return " + val + " : " + type_str);
}

}  // namespace codegen
}  // namespace pypto
