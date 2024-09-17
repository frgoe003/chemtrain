//
// Created by Paul Fuchs on 13.09.24.
//
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include "pybind11/embed.h"

#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>

#include "compile.h"
#include "xla_call_module_loader.h"

#include "xla/client/xla_computation.h"
#include "xla/service/hlo_parser.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // Include the Func dialect
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"


#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/transforms/Passes.h"

#include "tsl/platform/protobuf.h"
#include "tsl/platform/logging.h"


namespace py = pybind11;

namespace jcn {

    mlir::OwningOpRef<mlir::ModuleOp> ParseMlirModuleString(
            absl::string_view mlir_module_str, mlir::MLIRContext& context) {

}

    Compiler::Compiler(std::string py_executable) {

        std::cout << "Initialize the compiler" << std::endl;

        py::gil_scoped_acquire acquire;

        try {

            // TODO: We might wan to move these operations into another file
            //       sooner or later.
            py::exec(R"(
            import importlib
            import sys

            import jax
            import jax.numpy as jnp
            from jax import export

            print(f'Start defining the interface class in env {sys.prefix}')

            class ForceModule():

                def __init__(self, file_path):
                    # Load the module

                    print(f'Start loading the force function from {file_path}')

                    try:
                        with jax.default_device(jax.devices('cpu')[0]):
                            spec = importlib.util.spec_from_file_location("ff", file_path)
                            module = importlib.util.module_from_spec(spec)
                            sys.modules["ff"] = module
                            spec.loader.exec_module(module)
                    except Exception as e:
                        print(f'Failed to load the module: {e}')
                        raise e

                    print('Finished loading the force function')

                    n_atoms, max_neighbors = export.symbolic_shape(
                        "n_atoms, max_neighbors")

                    # n_atoms = 10
                    # max_neighbors = 9

                    position = jax.ShapeDtypeStruct((n_atoms, 3), jnp.float32)
                    neighbor = jax.ShapeDtypeStruct((n_atoms, max_neighbors), jnp.int32)

                    # Import the force function
                    self.force_fn: export.Exported = export.export(
                        jax.jit(module.force_fn), platforms=['cuda']
                    )(position, neighbor)

                    print("In and out avals")
                    print(self.force_fn.in_avals)
                    print(self.force_fn.out_avals)

                    print("Return traced model")

                    # bytestring = "\n".join([
                    #     line for line in self.force_fn.mlir_module().splitlines()
                    #     if not line.lstrip().startswith("stablehlo.custom_call")
                    # ])

                    bytestring = self.force_fn.mlir_module()

                    with open("module.mlir", "w") as f:
                        f.write(bytestring)

                    with open("module.mlir", "r") as f:
                        self.bytestring = f.read()


                def compile(self, n_atoms, max_neighbors):
                    # To avoid specifying data, we use only a ShapedArray for
                    # lowering

                    try:
                        return self.force_fn.serialize()
                    except Exception as e:
                        print(f'Failed to serialize the force function: {e}')
                        return ''

            )");

            std::cout << "Get the class" << std::endl;

            py::object ForceModule = py::globals()["ForceModule"];

            std::cout << "Instanciate it" << std::endl;

            // We import the lowering class using pybind
            py::object force_compiler = ForceModule(py_executable);

            std::cout << "Get the bytestring" << std::endl;
            bytestring = force_compiler.attr("bytestring").cast<std::string>();

            std::cout << "Module: " << bytestring << std::endl;

        } catch (const py::error_already_set& e) {
            std::cerr << "Python error: " << e.what() << std::endl;
        }

        std::cout << "Finished initialization" << std::endl;
    }

    xla::XlaComputation Compiler::compile(
        const int n_atoms, const int max_neighbor, mlir::MLIRContext& context) {
//        xla::HloSnapshot proto;
//        std::unique_ptr<xla::HloModule> module;
//
//
////        std::string bytestring;
////        try {
////            // We import the decorated module and get the serialized HLO module as a
////            // bytestring
////            py::gil_scoped_acquire acquire;
////
////            std::cout << "Start compiling the force function" << std::endl;
////            bytestring = force_compiler.attr(
////                "compile")(n_atoms, max_neighbor).cast<std::string>();
////
////        } catch (const py::error_already_set& e) {
////            std::cerr << "Python error: " << e.what() << std::endl;
////        }
//
//        // We try to serialize the protobuffer
//        if (!proto.ParseFromString(bytestring) &&
//            !proto.mutable_hlo()->ParseFromString(bytestring) &&
//            !proto.mutable_hlo()->mutable_hlo_module()->ParseFromString(bytestring)) {
//            return tsl::errors::InvalidArgument("Failed to parse input as HLO protobuf binary");
//        }

        // xla::XlaComputation computation;
        // absl::Status load_status = xla::ParseMlirModuleStringAndConvertToXlaComputation(
        //    bytestring, computation, false, false);

        std::cout << "Load dialects" << std::endl;

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

        std::cout << "Start parsing the bytestring" << std::endl;

        xla::Shape position_shape = xla::ShapeUtil::MakeShape(xla::F32, absl::Span<const int64_t>{10, 3});
        xla::Shape neighbor_shape = xla::ShapeUtil::MakeShape(xla::S32, absl::Span<const int64_t>{10, 9});

        std::vector<xla::Shape> inputShapes = {position_shape, neighbor_shape};

        std::cout << "Start loading the module" << std::endl;

        // std::vector<std::string> disabled_checks = {"shape_assertions"};
        std::vector<std::string> disabled_checks = {};
        std::vector<std::string> platforms = {"cuda"};
        std::unique_ptr<XlaCallModuleLoader> module_loader = XlaCallModuleLoader::Create(
            &context, 8, bytestring, disabled_checks, platforms, 2, false).value();

        absl::Status status;

        std::cout << "Validate dialect" << std::endl;

        status = module_loader->ValidateDialect();
        if (!status.ok()) {
            std::cerr << "Failed to validate dialect: " << status.message() << std::endl;
        }

        std::cout << "Start setting the platform index" << std::endl;

        status = module_loader->SetPlatformIndex("cuda");
        if (!status.ok()) {
            std::cerr << "Failed to set platform index: " << status.message() << std::endl;
        }

        std::cout << "Start refining the dynamic shapes" << std::endl;

        status = module_loader->RefineDynamicShapes(inputShapes);
        if (!status.ok()) {
            std::cerr << "Failed to refine dynamic shapes: " << status.message() << std::endl;
        }

        std::cout << "Start lowering the module to MHLO" << std::endl;

        status = module_loader->LowerModuleToMhlo();
        if (!status.ok()) {
            std::cerr << "Failed to refine dynamic shapes: " << status.message() << std::endl;
        }

        std::cout << "Finished preparing the XLA computation" << std::endl;

        auto res = module_loader->ToXlaComputation();
        if (!res.ok()) {
            std::cerr << "Failed to convert the module to XLA computation: " << res.status().message() << std::endl;
        }

        return std::move(res).value();

    }


} // namespace jcn
