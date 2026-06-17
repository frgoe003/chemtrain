/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================================================

Reproduced from https://github.com/tensorflow/tensorflow

*/

#include "xla_call_module_loader.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "shardy/dialect/sdy/ir/dialect.h"  // from @shardy
#include "shardy/dialect/sdy/transforms/import/passes.h"  // from @shardy
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "stablehlo/transforms/StablehloRefineShapes.h"  // from @stablehlo
// #include "tensorflow/compiler/jit/flags.h"
// #include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
// #include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/python/refine_polymorphic_shapes.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace jcn {

namespace {

// When adding a new version, write when it was added. Also change the default
// version in the constructor in xla.py.
// See
// https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#native-serialization-versions
// for a description of the different versions.

constexpr int kVersionStartStableHloCompatibility = 4;
constexpr int kVersionStartSupportCallTFGraph = 5;
constexpr int kVersionStartSupportDisabledChecks = 6;
constexpr int kVersionStartSupportShapeAssertions = 7;
constexpr int kVersionStartSupportUsesShapePolymorphismAttr = 8;
constexpr int kVersionStartSupportEffects = 9;
constexpr int kVersionStartSupportShardyPartitioner = 10;
constexpr int kVersionMinimumSupported = kVersionStartStableHloCompatibility;

// This should match xla.py:call_module_maximum_supported_version
constexpr int kVersionMaximumSupported = kVersionStartSupportShardyPartitioner;

constexpr llvm::StringRef kDisabledCheckPlatform = "platform";

bool IsPlatformCheckDisabled(absl::Span<const std::string> disabled_checks) {
  return llvm::is_contained(disabled_checks, kDisabledCheckPlatform);
}

constexpr llvm::StringRef kDisabledCheckShapeAssertions = "shape_assertions";

bool IsShapeAssertionsCheckDisabled(
    absl::Span<const std::string> loading_disabled_checks) {
  return llvm::is_contained(loading_disabled_checks,
                            kDisabledCheckShapeAssertions);
}

constexpr llvm::StringRef kUsesShapePolymorphismAttr =
    "jax.uses_shape_polymorphism";

constexpr llvm::StringRef kStablehloConvert = "stablehlo.convert";
constexpr llvm::StringRef kStablehloConstant = "stablehlo.constant";

struct ShapePromotionStats {
  int dynamic_reshape_i32_before = 0;
  int dynamic_reshape_i64_before = 0;
  int dynamic_broadcast_i32_before = 0;
  int dynamic_broadcast_i64_before = 0;
  int dynamic_iota_i32_before = 0;
  int dynamic_iota_i64_before = 0;
  int promoted_dynamic_operands = 0;
  int bypassed_i64_to_i32_narrowings = 0;
  int inserted_fallback_converts = 0;
  int cloned_shape_ops = 0;
};

bool DebugShapePromotionEnabled() {
  const char *value = std::getenv("CHEMTRAIN_DEBUG_SHAPE_PROMOTION");
  return value != nullptr && llvm::StringRef(value) != "0" &&
         llvm::StringRef(value).lower() != "false";
}

mlir::RankedTensorType RankedIntegerTensorType(mlir::Type type) {
  auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!ranked) return {};
  if (!ranked.getElementType().isSignlessInteger()) return {};
  return ranked;
}

bool IsRankedI32Tensor(mlir::Type type) {
  auto ranked = RankedIntegerTensorType(type);
  return ranked && ranked.getElementType().isSignlessInteger(32);
}

bool IsRankedI64Tensor(mlir::Type type) {
  auto ranked = RankedIntegerTensorType(type);
  return ranked && ranked.getElementType().isSignlessInteger(64);
}

mlir::RankedTensorType I64TensorLike(mlir::Type type) {
  auto ranked = RankedIntegerTensorType(type);
  if (!ranked) return {};
  return mlir::RankedTensorType::get(
      ranked.getShape(), mlir::IntegerType::get(type.getContext(), 64));
}

std::optional<int> DynamicShapeOperandIndex(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name == "stablehlo.dynamic_reshape") return 1;
  if (name == "stablehlo.dynamic_broadcast_in_dim") return 1;
  if (name == "stablehlo.dynamic_iota") return 0;
  return std::nullopt;
}

void CountDynamicShapeOperand(mlir::Operation *op, ShapePromotionStats *stats) {
  std::optional<int> shape_operand_index = DynamicShapeOperandIndex(op);
  if (!shape_operand_index) return;
  if (op->getNumOperands() <= *shape_operand_index) return;

  mlir::Type type = op->getOperand(*shape_operand_index).getType();
  bool is_i32 = IsRankedI32Tensor(type);
  bool is_i64 = IsRankedI64Tensor(type);
  llvm::StringRef name = op->getName().getStringRef();
  if (name == "stablehlo.dynamic_reshape") {
    stats->dynamic_reshape_i32_before += is_i32 ? 1 : 0;
    stats->dynamic_reshape_i64_before += is_i64 ? 1 : 0;
  } else if (name == "stablehlo.dynamic_broadcast_in_dim") {
    stats->dynamic_broadcast_i32_before += is_i32 ? 1 : 0;
    stats->dynamic_broadcast_i64_before += is_i64 ? 1 : 0;
  } else if (name == "stablehlo.dynamic_iota") {
    stats->dynamic_iota_i32_before += is_i32 ? 1 : 0;
    stats->dynamic_iota_i64_before += is_i64 ? 1 : 0;
  }
}

mlir::Attribute PromoteIntegerConstantAttrToI64(mlir::Attribute attr,
                                                mlir::RankedTensorType type) {
  auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr);
  if (!dense) return {};

  mlir::RankedTensorType promoted_type = I64TensorLike(type);
  if (!promoted_type) return {};

  llvm::SmallVector<int64_t> values;
  values.reserve(dense.getNumElements());
  for (llvm::APInt value : dense.getValues<llvm::APInt>()) {
    values.push_back(value.getSExtValue());
  }
  return mlir::DenseIntElementsAttr::get(promoted_type, values);
}

mlir::Value CreateConvertToI64(mlir::Value value, mlir::OpBuilder *builder,
                               ShapePromotionStats *stats) {
  mlir::RankedTensorType promoted_type = I64TensorLike(value.getType());
  if (!promoted_type) return {};

  mlir::OperationState state(value.getLoc(), kStablehloConvert);
  state.addOperands(value);
  state.addTypes(promoted_type);
  mlir::Operation *converted = builder->create(state);
  ++stats->inserted_fallback_converts;
  return converted->getResult(0);
}

bool IsCloneableShapeOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "stablehlo.reshape" || name == "stablehlo.concatenate" ||
         name == "stablehlo.add" || name == "stablehlo.multiply" ||
         name == "stablehlo.subtract" || name == "stablehlo.maximum" ||
         name == "stablehlo.minimum";
}

void DumpShapeProducerChain(mlir::Value value, int depth) {
  if (depth > 8) {
    LOG(ERROR) << "    ... producer chain truncated";
    return;
  }

  mlir::Operation *producer = value.getDefiningOp();
  if (producer == nullptr) {
    LOG(ERROR) << "    depth=" << depth
               << " block_argument type=" << mlir::debugString(value.getType())
               << " loc=" << mlir::debugString(value.getLoc());
    return;
  }

  LOG(ERROR) << "    depth=" << depth
             << " op=" << producer->getName().getStringRef().str()
             << " result_type=" << mlir::debugString(value.getType())
             << " loc=" << mlir::debugString(producer->getLoc());
  for (mlir::Value operand : producer->getOperands()) {
    if (RankedIntegerTensorType(operand.getType())) {
      DumpShapeProducerChain(operand, depth + 1);
    }
  }
}

mlir::Value PromoteShapeValueToI64(
    mlir::Value value, mlir::OpBuilder *builder,
    llvm::DenseMap<mlir::Value, mlir::Value> *cache,
    ShapePromotionStats *stats) {
  if (IsRankedI64Tensor(value.getType())) return value;
  if (!IsRankedI32Tensor(value.getType())) return {};

  auto cached = cache->find(value);
  if (cached != cache->end()) return cached->second;

  mlir::Operation *producer = value.getDefiningOp();
  if (producer == nullptr) {
    mlir::Value promoted = CreateConvertToI64(value, builder, stats);
    (*cache)[value] = promoted;
    return promoted;
  }

  llvm::StringRef name = producer->getName().getStringRef();
  if (name == kStablehloConvert && producer->getNumOperands() == 1) {
    mlir::Value source = producer->getOperand(0);
    if (IsRankedI64Tensor(source.getType())) {
      ++stats->bypassed_i64_to_i32_narrowings;
      (*cache)[value] = source;
      return source;
    }
    mlir::Value promoted_source =
        PromoteShapeValueToI64(source, builder, cache, stats);
    if (promoted_source) {
      (*cache)[value] = promoted_source;
      return promoted_source;
    }
  }

  if (name == kStablehloConstant) {
    mlir::RankedTensorType promoted_type = I64TensorLike(value.getType());
    mlir::Attribute value_attr = producer->getAttr("value");
    mlir::Attribute promoted_attr =
        PromoteIntegerConstantAttrToI64(value_attr, promoted_type);
    if (promoted_attr) {
      mlir::OperationState state(producer->getLoc(), kStablehloConstant);
      state.addAttribute("value", promoted_attr);
      state.addTypes(promoted_type);
      mlir::Operation *constant = builder->create(state);
      ++stats->cloned_shape_ops;
      (*cache)[value] = constant->getResult(0);
      return constant->getResult(0);
    }
  }

  if (IsCloneableShapeOp(producer)) {
    llvm::SmallVector<mlir::Value> promoted_operands;
    promoted_operands.reserve(producer->getNumOperands());
    for (mlir::Value operand : producer->getOperands()) {
      mlir::Value promoted_operand =
          PromoteShapeValueToI64(operand, builder, cache, stats);
      if (!promoted_operand) return CreateConvertToI64(value, builder, stats);
      promoted_operands.push_back(promoted_operand);
    }

    mlir::RankedTensorType promoted_type = I64TensorLike(value.getType());
    if (!promoted_type) return {};
    mlir::OperationState state(producer->getLoc(),
                               producer->getName().getStringRef());
    state.addOperands(promoted_operands);
    state.addAttributes(producer->getAttrs());
    state.addTypes(promoted_type);
    mlir::Operation *cloned = builder->create(state);
    ++stats->cloned_shape_ops;
    (*cache)[value] = cloned->getResult(0);
    return cloned->getResult(0);
  }

  mlir::Value promoted = CreateConvertToI64(value, builder, stats);
  (*cache)[value] = promoted;
  return promoted;
}

ShapePromotionStats PromoteDynamicShapeOperandsToI64(mlir::ModuleOp module) {
  ShapePromotionStats stats;
  module.walk([&](mlir::Operation *op) { CountDynamicShapeOperand(op, &stats); });

  const bool verbose = DebugShapePromotionEnabled();
  llvm::SmallVector<mlir::Operation *> dynamic_ops;
  module.walk([&](mlir::Operation *op) {
    if (DynamicShapeOperandIndex(op)) dynamic_ops.push_back(op);
  });

  for (mlir::Operation *op : dynamic_ops) {
    std::optional<int> shape_operand_index = DynamicShapeOperandIndex(op);
    if (!shape_operand_index || op->getNumOperands() <= *shape_operand_index) {
      continue;
    }
    mlir::Value shape_operand = op->getOperand(*shape_operand_index);
    if (!IsRankedI32Tensor(shape_operand.getType())) continue;

    mlir::OpBuilder builder(op);
    llvm::DenseMap<mlir::Value, mlir::Value> cache;
    mlir::Value promoted =
        PromoteShapeValueToI64(shape_operand, &builder, &cache, &stats);
    if (!promoted || !IsRankedI64Tensor(promoted.getType())) continue;

    op->setOperand(*shape_operand_index, promoted);
    ++stats.promoted_dynamic_operands;

    if (verbose) {
      LOG(INFO) << "Promoted StableHLO dynamic shape operand: op="
                << op->getName().getStringRef().str()
                << " loc=" << mlir::debugString(op->getLoc())
                << " old_type=" << mlir::debugString(shape_operand.getType())
                << " new_type=" << mlir::debugString(promoted.getType());
    } else {
      VLOG(2) << "Promoted StableHLO dynamic shape operand: op="
              << op->getName().getStringRef().str()
              << " loc=" << mlir::debugString(op->getLoc())
              << " old_type=" << mlir::debugString(shape_operand.getType())
              << " new_type=" << mlir::debugString(promoted.getType());
    }
  }

  std::string summary = absl::StrCat(
      "StableHLO dynamic shape promotion summary: promoted=",
      stats.promoted_dynamic_operands, " cloned_shape_ops=",
      stats.cloned_shape_ops, " bypassed_i64_to_i32_narrowings=",
      stats.bypassed_i64_to_i32_narrowings, " fallback_converts=",
      stats.inserted_fallback_converts, " dynamic_reshape_i32_before=",
      stats.dynamic_reshape_i32_before, " dynamic_reshape_i64_before=",
      stats.dynamic_reshape_i64_before, " dynamic_broadcast_i32_before=",
      stats.dynamic_broadcast_i32_before, " dynamic_broadcast_i64_before=",
      stats.dynamic_broadcast_i64_before, " dynamic_iota_i32_before=",
      stats.dynamic_iota_i32_before, " dynamic_iota_i64_before=",
      stats.dynamic_iota_i64_before);
  if (verbose) {
    LOG(INFO) << summary;
  } else {
    VLOG(1) << summary;
  }
  return stats;
}

void DumpRemainingI32DynamicShapeOperands(mlir::ModuleOp module) {
  int remaining = 0;
  module.walk([&](mlir::Operation *op) {
    std::optional<int> shape_operand_index = DynamicShapeOperandIndex(op);
    if (!shape_operand_index || op->getNumOperands() <= *shape_operand_index) {
      return;
    }
    mlir::Value shape_operand = op->getOperand(*shape_operand_index);
    if (!IsRankedI32Tensor(shape_operand.getType())) return;
    ++remaining;
    LOG(ERROR) << "Remaining i32 StableHLO dynamic shape operand: op="
               << op->getName().getStringRef().str()
               << " loc=" << mlir::debugString(op->getLoc())
               << " shape_type=" << mlir::debugString(shape_operand.getType());
    DumpShapeProducerChain(shape_operand, 0);
  });
  LOG(ERROR) << "Remaining i32 StableHLO dynamic shape operands: " << remaining;
}

}  // namespace

bool IsTokenType(mlir::Type type) {
  return mlir::isa<mlir::stablehlo::TokenType>(type);
}

absl::StatusOr<std::unique_ptr<XlaCallModuleLoader>>
XlaCallModuleLoader::Create(mlir::MLIRContext *context, int version,
                            mlir::StringRef module_str,
                            std::vector<std::string> disabled_checks,
                            std::vector<std::string> platforms,
                            int num_invocation_args,
                            bool main_has_token_input_output,
                            bool use_shardy_partitioner) {
  std::unique_ptr<XlaCallModuleLoader> loader(new XlaCallModuleLoader);
  TF_RETURN_IF_ERROR(loader->LoadModule(
      context, version, module_str, std::move(disabled_checks),
      std::move(platforms), num_invocation_args, main_has_token_input_output,
      use_shardy_partitioner));
  return loader;
}

absl::Status XlaCallModuleLoader::SetPlatformIndex(
    absl::string_view compilation_platform) {
  int platform_index = -1;
  if (!platforms_.empty()) {
    auto found_platform =
        std::find(platforms_.begin(), platforms_.end(), compilation_platform);
    if (found_platform == platforms_.end()) {
      if (!IsPlatformCheckDisabled(loading_disabled_checks_)) {
        return absl::NotFoundError(absl::StrCat(
            "The current platform ", compilation_platform,
            " is not among the platforms required by the module: [",
            absl::StrJoin(platforms_, ", "), "]"));
      } else {
        if (platforms_.size() > 1) {
          platform_index = 0;
        }
      }
    } else {
      // We only use a platform index argument if we support at least 2
      // platforms.
      if (platforms_.size() > 1) {
        platform_index = found_platform - platforms_.begin();
      }
    }
  }

  if (platform_index < 0) return absl::OkStatus();
  VLOG(3) << "XlaCallModule setting the platform_index to " << platform_index
          << " for platform " << compilation_platform << ".";
  mlir::Block &main_body = main_.front();

  if (main_.getNumArguments() < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The module should have a platform index argument but it has no ",
        "arguments"));
  }
  mlir::OpBuilder op_builder(main_);
  op_builder.setInsertionPointToStart(&main_body);
  mlir::BlockArgument platform_index_arg = main_body.getArgument(0);
  mlir::RankedTensorType arg_ranked_type =
      mlir::dyn_cast<mlir::RankedTensorType>(platform_index_arg.getType());
  if (!arg_ranked_type || arg_ranked_type.getRank() != 0 ||
      !(arg_ranked_type.getElementType().isSignlessInteger(32) ||
        arg_ranked_type.getElementType().isSignlessInteger(64))) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module argument at index 0 should be a 0-dimensional "
                     "32-bit or 64-bit integer-tensor platform index argument "
                     "but has type ",
                     mlir::debugString(platform_index_arg.getType())));
  }
  bool is_32_bit = arg_ranked_type.getElementType().isSignlessInteger(32);
  auto const_attr = is_32_bit ? op_builder.getI32IntegerAttr(platform_index)
                              : op_builder.getI64IntegerAttr(platform_index);
  auto platform_index_op = op_builder.create<mlir::stablehlo::ConstantOp>(
      platform_index_arg.getLoc(), const_attr);
  platform_index_arg.replaceAllUsesWith(platform_index_op);

  CHECK(llvm::succeeded(main_.eraseArgument(0)));
  platform_index_arg_set_ = true;
  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::RefineDynamicShapes(
    llvm::ArrayRef<xla::Shape> input_shapes) {
  // Skip shape refinement for new versions if USES_SHAPE_POLYMORPHISM_ATTR=1
  if (version_ >= kVersionStartSupportUsesShapePolymorphismAttr) {
    if (mlir::Attribute uses_shape_poly_attr =
            (*module_)->getAttr(kUsesShapePolymorphismAttr)) {
      mlir::BoolAttr uses_shape_poly_bool_attr =
          llvm::dyn_cast<mlir::BoolAttr>(uses_shape_poly_attr);

      if (!uses_shape_poly_bool_attr) {
        return absl::InvalidArgumentError(absl::StrCat(
            "jax.uses_shape_polymorphism is not a boolean attribute: ",
            mlir::debugString(uses_shape_poly_attr)));
      }
      if (!uses_shape_poly_bool_attr.getValue()) {
        VLOG(3) << "XlaCallModule skipping shape refinement due to module "
                << " attribute " << kUsesShapePolymorphismAttr.str() << "="
                << mlir::debugString(uses_shape_poly_attr);
        return absl::OkStatus();
      }
    } else {
      VLOG(3) << "XlaCallModule skipping shape refinement due to module "
              << " attribute " << kUsesShapePolymorphismAttr.str()
              << " missing";
      return absl::OkStatus();
    }
  }
  // Add the tokens to the input_shapes. Starting with version 9, the main
  // function may take token arguments that do not correspond with op inputs.
  int nr_inputs = NrInputs();
  int nr_expected_tokens = llvm::count_if(InputTypes(), IsTokenType);
  bool has_platform_index_arg =
      platforms_.size() > 1 && !platform_index_arg_set_;
  int nr_expected_platform_index_args = has_platform_index_arg ? 1 : 0;
  if (input_shapes.size() !=
      nr_inputs - nr_expected_tokens - nr_expected_platform_index_args) {
    return absl::InvalidArgumentError(absl::StrCat(
        "XlaCallModule RefineDynamicShapes called with ", input_shapes.size(),
        " input shapes, but the main function takes ",
        nr_inputs - nr_expected_tokens - nr_expected_platform_index_args,
        " non-token and non-platform-index arguments. The input ",
        "shapes are (",
        absl::StrJoin(input_shapes, ", ",
                      [](std::string *out, const xla::Shape &s) {
                        absl::StrAppend(out, s.ToString());
                      }),
        ") and the main function argument types are ",
        absl::StrJoin(InputTypes(), ", ",
                      [](std::string *out, const mlir::Type &t) {
                        absl::StrAppend(out, mlir::debugString(t));
                      }),
        ")"));
  }

  // Derive static input types to use for main.
  mlir::Block &main_body = main_.front();
  mlir::Builder builder(module_->getContext());
  std::vector<mlir::Type> static_array_input_types(nr_inputs);
  int next_actual_input = 0;
  for (int i = 0, end = nr_inputs; i < end; ++i) {
    mlir::Type arg_type = main_body.getArgument(i).getType();
    if (i == 0 && has_platform_index_arg) {
      static_array_input_types[i] = arg_type;
      continue;
    }
    if (IsTokenType(arg_type)) {
      static_array_input_types[i] = arg_type;
      VLOG(3) << "XlaCallModule static array input type #" << i << ": "
              << mlir::debugString(static_array_input_types[i])
              << " for argument type " << mlir::debugString(arg_type);
      continue;
    }

    // Get static MLIR Type from xla Shape.
    const xla::Shape &xla_shape = input_shapes[next_actual_input++];
    std::vector<int64_t> xla_dimensions;
    if (xla_shape.IsArray()) {
      xla_dimensions = std::vector<int64_t>(xla_shape.dimensions().begin(),
                                            xla_shape.dimensions().end());
    }
    TF_ASSIGN_OR_RETURN(
        mlir::Type element_type,
        ConvertPrimitiveTypeToMlirType(xla_shape.element_type(), builder));
    mlir::RankedTensorType type =
        mlir::RankedTensorType::get(xla_dimensions, element_type);

    VLOG(3) << "XlaCallModule static array input type #" << i << ": "
            << mlir::debugString(type) << " for argument type "
            << mlir::debugString(arg_type);
    static_array_input_types[i] = type;
  }

  // Insert custom_call ops as shims to maintain the validity of the module when
  // main's input types are changed later. This is a workaround to allow shape
  // refinement to be applied; the custom_calls are removed before returning.
  // Arguments to main may occur as return values, or as inputs to called
  // functions, and changing their types may invalidate the module due to type
  // mismatches. To prevent this, for each argument that is a dynamically-shaped
  // tensor, we insert a custom_call op that takes the argument as an input and
  // replace uses of the argument with the custom_call's result. custom_call
  // is used as it allows its inputs and outputs to be unranked.
  //
  // Example:
  //
  // The below main function returns its argument directly:
  //
  // func.func @main(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  //   return %arg0 : tensor<*xf32>
  // }
  //
  // Changing the argument's type invalidates the IR (type mismatch):
  //
  // func.func @main(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  //   return %arg0 : tensor<*xf32>
  // }
  //
  // Inserting a custom_call allows the IR to remain valid:
  //
  // func.func @main(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  //   %0 = stablehlo.constant dense<[2, 3]> : tensor<2xi64>
  //   %1 = stablehlo.custom_call
  //   @stablehlo.shape_refinement_operand_wrapper(%arg0, %0)
  //   {indices_of_shape_operands = dense<1> : tensor<1xi64>} :
  //   (tensor<2x3xf32>, tensor<2xi64>) -> tensor<*xf32>
  //   return %1 : tensor<*xf32>
  // }
  //
  // After shapes are refined and the custom_calls are removed, we get:
  //
  // func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  //   return %arg0 : tensor<2x3xf32>
  // }
  //
  {
    if (failed(mlir::stablehlo::refineArguments(main_,
                                                static_array_input_types))) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error refining argument shapes."));
    }
  }

  PromoteDynamicShapeOperandsToI64(*module_);
  if (mlir::failed(mlir::verify(*module_))) {
    DumpRemainingI32DynamicShapeOperands(*module_);
    return absl::InvalidArgumentError(
        absl::StrCat("Error verifying module after promoting StableHLO dynamic "
                     "shape operands to i64."));
  }

  bool enable_shape_assertions =
      (version_ >= kVersionStartSupportShapeAssertions &&
       !IsShapeAssertionsCheckDisabled(loading_disabled_checks_));

  // Store the original output types before shape refinement.
  mlir::TypeRange original_output_types = OutputTypes();

  // RefinePolymorphicShapes will refine using the new static types and clean up
  // the shape_refinement_operand_wrapper custom calls.
  absl::Status refine_status =
      xla::RefinePolymorphicShapes(*module_, enable_shape_assertions);
  if (!refine_status.ok()) {
    DumpRemainingI32DynamicShapeOperands(*module_);
    return refine_status;
  }

  // Mark the output types as refined if they are different from the original
  // output types.
  if (OutputTypes() != original_output_types) {
    output_types_refined_ = true;
  }

  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::LoadModule(
    mlir::MLIRContext *context, int version, mlir::StringRef module_str,
    std::vector<std::string> disabled_checks,
    std::vector<std::string> platforms, int num_invocation_args,
    bool main_has_token_input_output, bool use_shardy_partitioner) {
  context_ = context;
  version_ = version;
  platforms_ = platforms;
  loading_disabled_checks_ = disabled_checks;
  use_shardy_partitioner_ = use_shardy_partitioner;

  // Load a superset of dialects; we should check at serialization time that
  // we only include allowable dialects.
  context_->loadDialect<mlir::func::FuncDialect>();
  context_->loadDialect<mlir::stablehlo::StablehloDialect>();
  context_->loadDialect<mlir::chlo::ChloDialect>();
  context_->loadDialect<mlir::vhlo::VhloDialect>();
  context_->loadDialect<mlir::sdy::SdyDialect>();

  if (version < kVersionMinimumSupported) {
    return absl::InvalidArgumentError(absl::StrCat(
        "XlaCallModuleOp with version ", version,
        " is not supported anymore. Must be >= ", kVersionMinimumSupported));
  }
  if (version > kVersionMaximumSupported) {
    return absl::InvalidArgumentError(
        absl::StrCat("XlaCallModuleOp with version ", version,
                     " is not supported by this build. Must be <= ",
                     kVersionMaximumSupported));
  }
  if (version >= kVersionStartSupportDisabledChecks && platforms.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("XlaCallModuleOp with version ", version,
                     " must have non-empty platforms."));
  }

  // Parse the StableHLO/VHLO bytecode
  {
    module_ =
        mlir::stablehlo::deserializePortableArtifact(module_str, context_);
    if (!module_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot deserialize computation."));
    }
  }

  if (use_shardy_partitioner) {
    // We need to inline `sdy.mesh` symbols because otherwise they are going
    // to be discarded or their names might collide with `sdy.mesh` symbols in
    // another XlaCallModuleOp.
    mlir::PassManager pm(module_->getContext());
    // TODO(b/422690222): Remove `addSdyRoundTripImportPipeline` 6 months
    // after mixed serialization will be supported by Shardy+StableHLO in JAX
    xla::sdy::addSdyRoundTripImportPipeline(pm, /*enableConstantImport=*/false);
    pm.addPass(mlir::sdy::createInlineMeshesPass());
    if (failed(pm.run(*module_))) {
      return absl::InternalError(
          absl::StrCat("Shardy inline meshes pass failed. "));
    }
  }

  {
    if (mlir::failed(mlir::verify(*module_))) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Error verifying module."));
    }
  }
  main_ = module_->lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_) {
    return absl::InvalidArgumentError("Cannot find 'main' in module");
  }

  mlir::Block &main_body = main_.front();

  int nr_token_arguments = llvm::count_if(InputTypes(), IsTokenType);
  if (version < kVersionStartSupportEffects) {
    bool has_token_at_start = (nr_token_arguments == 1 &&
                               IsTokenType(main_.getArgument(0).getType()));
    if (main_has_token_input_output != has_token_at_start) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected a token at start iff main_has_token_input_output. ",
          "Found main function type ",
          mlir::debugString(main_.getFunctionType()),
          " and main_has_token_input_output = ", main_has_token_input_output));
    }
  }
  int nr_platform_args = (platforms.size() > 1 ? 1 : 0);
  if (num_invocation_args !=
      main_body.getNumArguments() - nr_platform_args - nr_token_arguments) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Incorrect number of arguments passed to XlaCallModule = ",
        num_invocation_args, ". It must be called with ",
        main_body.getNumArguments() - nr_platform_args - nr_token_arguments,
        " because the module main function takes ", main_body.getNumArguments(),
        " arguments of which ", nr_platform_args, " platform index arguments, ",
        "and ", nr_token_arguments, " token arguments."));
  }
  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::ValidateXlaCallModuleInvariants() {
  bool moduleValidationFailed = false;

  module_->walk([&](mlir::Operation *op) {
    // StableHLO programs created by jax2tf only contain operations
    // from Builtin, Func, StableHLO, Shardy dialects.
    if (!llvm::isa<mlir::BuiltinDialect, mlir::chlo::ChloDialect,
                   mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect,
                   mlir::sdy::SdyDialect>(op->getDialect())) {
      op->emitOpError() << "is an op from an unsupported dialect";
      moduleValidationFailed = true;
    }
    // `shape_assertion` custom calls must have side effects. We check this here
    // because a pure `shape_assertion` is likely to be removed by MLIR's
    // dead-code elimination, preventing us from detecting the issue later.
    if (auto customCallOp = llvm::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
      if (!customCallOp.getHasSideEffect() &&
          customCallOp.getCallTargetName() == "shape_assertion") {
        op->emitOpError() << "`shape_assertion` custom calls must set "
                             "`has_side_effect = true`.";
        moduleValidationFailed = true;
      }
    }
  });

  if (moduleValidationFailed) {
    return absl::InvalidArgumentError(
        absl::StrCat("XlaCallModule failed validation."));
  }
  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::ValidateStaticShapes() {
  return xla::ValidateStaticShapes(*module_);
}

absl::Status XlaCallModuleLoader::PrepareStablehloForLowering() {

  // TODO (b/410057228): Replace MHLO canonicalization with StableHLO.
  // This code requires MHLO CaseOp canonicalization to remove unreachable
  // branches, else `tf.call_tf_function` inlining can fail.
  mlir::PassManager pm(module_->getContext());
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (use_shardy_partitioner_) {
    // We need to export shardings because the lowering path go directly to
    // HLO but not the MLIR to HLO path that invokes SdyRoundTripExport.
    // We keep meshes inlined to avoid naming collisions when multiple
    // XlaCallModules are combined.
    xla::sdy::addSdyRoundTripExportPipeline(pm);
  }

  if (failed(pm.run(*module_))) {
    return absl::InternalError(
        absl::StrCat("MHLO->HLO lowering passes failed."));
  }

  return absl::OkStatus();
}

absl::StatusOr<xla::XlaComputation> XlaCallModuleLoader::ToXlaComputation() {
  xla::HloProto proto;
  TF_RETURN_IF_ERROR(xla::ConvertStablehloToHloProto(*module_, &proto));
  return xla::XlaComputation(std::move(*proto.mutable_hlo_module()));
}

}  // namespace jcn
