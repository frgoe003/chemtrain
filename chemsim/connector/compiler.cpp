//
// Created by Paul Fuchs on 13.09.24.
//

#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>

#include "compiler.h"
#include "xla_call_module_loader.h"

#include "xla/client/xla_computation.h"
#include "xla/service/hlo_parser.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"

#include "absl/types/span.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/MLIRContext.h"

#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"


namespace jcn {

    Compiler::Compiler(const std::string& mlir_module) : mlir_module(mlir_module) {

        // We add some dialects to interpret the MLIR module from JAX

        mlir::DialectRegistry registry;

        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::ml_program::MLProgramDialect>();
        registry.insert<mlir::shape::ShapeDialect>();
        mlir::func::registerAllExtensions(registry);
        mlir::mhlo::registerAllMhloDialects(registry);
        mlir::sdy::registerAllDialects(registry);
        mlir::stablehlo::registerAllDialects(registry);

        context.appendDialectRegistry(registry);

    }

    xla::XlaComputation Compiler::compile(
        const int n_atoms,
        std::vector<std::vector<int64_t>> graph_shapes,
        std::vector<xla::PrimitiveType> graph_types
        ) {

        // For shape refinement, we have to provide the shapes of the input tensors
        xla::Shape position_shape = xla::ShapeUtil::MakeShape(xla::F32, absl::Span<const int64_t>{n_atoms, 3});
        xla::Shape species_shape = xla::ShapeUtil::MakeShape(xla::S32, absl::Span<const int64_t>{n_atoms});
        xla::Shape num_shape = xla::ShapeUtil::MakeShape(xla::S32, absl::Span<const int64_t>{1});

        std::vector<xla::Shape> inputShapes = {
            position_shape, species_shape, num_shape, num_shape};

        // For different graphs, the input shapes can vary
        int input_args = 4;
        for (size_t i = 0; i < graph_shapes.size(); i++) {
            xla::Shape neighbor_shape = xla::ShapeUtil::MakeShape(
                graph_types[i], graph_shapes[i]);
            inputShapes.push_back(neighbor_shape);
            input_args++;
        }

        for (const auto& shape : inputShapes) {
            std::cout << "Shape: " << shape.ToString() << std::endl;
            std::cout << "Element Type: " << xla::PrimitiveType_Name(shape.element_type()) << std::endl;
        }

        std::cout << "Input arguments: " << input_args << std::endl;

        std::cout << "Start loading the module" << std::endl;
        std::vector<std::string> disabled_checks = {};
        std::vector<std::string> platforms = {"cuda"};

        std::unique_ptr<XlaCallModuleLoader> module_loader = XlaCallModuleLoader::Create(
            &context, 9, mlir_module, disabled_checks, platforms, input_args, false).value();

        // We now follow the steps as in the XLACallModuleLoader from tensorflow
        absl::Status status;

        std::cout << "Validate dialect" << std::endl;
        status = module_loader->ValidateDialect();
        if (!status.ok()) {
            throw std::runtime_error(
                "Failed to validate dialect: " + std::string(status.message())
            );
        }

        std::cout << "Start setting the platform index" << std::endl;
        status = module_loader->SetPlatformIndex("cuda");
        if (!status.ok()) {
            throw std::runtime_error(
              "Failed to set platform index: " + std::string(status.message())
            );
        }

        std::cout << "Start refining the dynamic shapes" << std::endl;
        status = module_loader->RefineDynamicShapes(inputShapes);
        if (!status.ok()) {
            throw std::runtime_error(
                "Failed to refine dynamic shapes: " + std::string(status.message())
            );
        }

        std::cout << "Start lowering the module to MHLO" << std::endl;
        status = module_loader->LowerModuleToMhlo();
        if (!status.ok()) {
            throw std::runtime_error(
                "Failed to refine dynamic shapes: " + std::string(status.message())
            );
        }

        std::cout << "Finished preparing the XLA computation" << std::endl;
        auto res = module_loader->ToXlaComputation();
        if (!res.ok()) {
            throw std::runtime_error(
                "Failed to convert the module to XLA computation: " + std::string(status.message())
            );
        }

        return std::move(res).value();
    }


} // namespace jcn
