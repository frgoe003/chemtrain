//
// Created by Paul Fuchs on 13.09.24.
//
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include "pybind11/embed.h"

#include <iostream>
#include <string>
#include <cstdlib>

#include "compile.h"

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


struct ShapeRefinementPass : public mlir::PassWrapper<ShapeRefinementPass, mlir::OperationPass<mlir::ModuleOp>> {
  // Constructor to accept shapes programmatically
  ShapeRefinementPass(std::vector<std::vector<int64_t>> inputShapes) : inputShapes(inputShapes) {}

  // Vector of vectors for shapes, where each vector represents the shape for an argument
  std::vector<std::vector<int64_t>> inputShapes;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Iterate over each function in the module
    module.walk([&](mlir::Operation *op) {
      if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {  // Cast Operation* to FuncOp in func dialect
        if (func.getName() == "main") { // Specifically target the 'main' function
          auto funcType = func.getFunctionType();
          llvm::errs() << "Processing function: " << func.getName() << "\n";

          // Get input types
          llvm::ArrayRef<mlir::Type> inputTypes = funcType.getInputs();
          llvm::SmallVector<mlir::Type, 4> refinedInputTypes;

          // Refine each input type
          for (size_t i = 0; i < inputTypes.size(); ++i) {
            auto argType = inputTypes[i].dyn_cast<mlir::RankedTensorType>();
            if (!argType) continue; // Skip non-tensor arguments

            // If dynamic shape, refine to provided concrete shape
            if (argType.hasStaticShape()) {
              // Static shape, keep it as is
              refinedInputTypes.push_back(argType);
            } else {
              if (i < inputShapes.size()) {
                auto concreteShape = inputShapes[i];
                auto elementType = argType.getElementType();
                auto refinedType = mlir::RankedTensorType::get(concreteShape, elementType);
                refinedInputTypes.push_back(refinedType);

                llvm::errs() << "Refining argument " << i << " to shape: ";
                for (auto dim : concreteShape) llvm::errs() << dim << " ";
                llvm::errs() << "\n";
              } else {
                llvm::errs() << "No shape provided for argument " << i << "\n";
                refinedInputTypes.push_back(argType);  // Keep original if no shape provided
              }
            }
          }

          // Create new function type with refined inputs and original outputs
          auto newFuncType = mlir::FunctionType::get(func.getContext(), refinedInputTypes, funcType.getResults());

          // Set the new type for the function
          func.setType(newFuncType);
        }
      }
    });
  }
};

// Register the pass
// static mlir::PassRegistration<ShapeRefinementPass> pass("shape-refinement", "Refine input shapes to concrete values.");



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
                        jax.jit(module.force_fn)# , platforms='cuda'
                    )(position, neighbor)

                    print("In and out avals")
                    print(self.force_fn.in_avals)
                    print(self.force_fn.out_avals)

                    print("Return traced model")

                    self.bytestring = self.force_fn.mlir_module()

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

            bytestring = force_compiler.attr("bytestring").cast<std::string>();

            // std::cout << "Module: " << bytestring << std::endl;

        } catch (const py::error_already_set& e) {
            std::cerr << "Python error: " << e.what() << std::endl;
        }

        std::cout << "Finished initialization" << std::endl;
    }

    mlir::OwningOpRef<mlir::ModuleOp> Compiler::compile(
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

        std::cout << "Start parsing the bytestring" << std::endl;
        mlir::BaseScopedDiagnosticHandler diag_handler(&context);

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

        // mlir::ScopedDiagnosticHandler diagnostic_handler(&context);
        mlir::OwningOpRef<mlir::ModuleOp> module =
          mlir::parseSourceString<mlir::ModuleOp>(
              llvm::StringRef(bytestring.data(), bytestring.size()),
              // IR may be invalid because some fields may be using DenseElements
              // instead of DenseArray. We rectify that below and verify after.
              mlir::ParserConfig{&context, /*verifyAfterParse=*/false});
        if (!module) {
        // return diagnostic_handler.ConsumeStatus();
        }

        std::vector<std::vector<int64_t>> inputShapes = {{n_atoms, 3}, {n_atoms, max_neighbor}};

        mlir::PassManager pm(&context);
        // pm.addPass(mlir::mhlo::createShapeInferencePass());
        pm.addPass(std::make_unique<ShapeRefinementPass>(inputShapes));
        pm.addPass(mlir::mhlo::createLegalizeHloToLinalgPass());
        pm.addPass(mlir::stablehlo::createStablehloRefineShapesPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
        pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::mhlo::createChloLegalizeToHloPass());
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::mhlo::createSinkConstantsToControlFlowPass());

        pm.addPass(mlir::createInlinerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::stablehlo_ext::createChloRecomposeOpsPass());
        pm.addPass(mlir::stablehlo_ext::createStablehloRefineShapesPass());
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::stablehlo_ext::createStablehloCanonicalizeDynamismPass());
        // pm.addNestedPass<mlir::func::FuncOp>(
        //     std::make_unique<CheckShapeAssertionsPass>(enable_shape_assertions));
        if (!mlir::succeeded(pm.run(*module))) {
            std::cout << absl::StrCat("Module shape refinement failed: ",
                             diag_handler.ConsumeStatus().ToString()) << std::endl;
        }

        // pm.addPass(mlir::mhlo::createShapeLegalizeToHloPass());
        // pm.addPass(mlir::stablehlo::createStablehloCanonicalizeDynamismPass());


        // In order to export to XLA, we must sink constants to control flow
        // regions, since XLA uses functional control flow.

        // In
        // https://github.com/google/jax/commit/184e3a88004680dbf34328b05c5fc0d869cc4a93,
        // fields on some ops were changed to use Dense{Bool,I64}ArrayAttr instead of
        // I64DenseElementsAttr (DenseIntElementsAttr). Some clients still expect
        // dense elements, not dense arrays, so when serializing we always convert the
        // arrays to elements. The elements need to be converted back to arrays when
        // deserializing.
        // TODO: b/320507168 - Remove the conversion code, and verifyAfterParse.
        // xla::TF_RETURN_IF_ERROR(UpgradeVersionedStablehlo(*module));
        if (failed(module->verifyInvariants())) {
        VLOG(1) << "MLIR verification failed.";
        //    module->dump();
        // return diagnostic_handler.ConsumeStatus();
        }

        std::cout << "Finished parsing the bytestring" << std::endl;

        return std::move(module);
    }


} // namespace jcn
