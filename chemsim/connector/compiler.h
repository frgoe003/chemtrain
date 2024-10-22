//
// Created by Paul Fuchs on 13.09.24.
//

#include <string>

#include "xla/service/hlo_parser.h"
#include "xla/client/xla_computation.h"

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

#include "absl/types/span.h"


#ifndef COMPILE_H
#define COMPILE_H


namespace jcn {
    class Compiler {
    public:
        Compiler(const std::string& mlir_module);
        ~Compiler() = default;

        /**
        * @brief Prepares the MLIR module for XLA by performing dynamic shape refinement.
        *
        * @param n_atoms The number of atoms in the system (including ghost atoms)
        *     and invalid atoms to be masked out. Determines also the size of
        *     the species array and the ghost mask.
        * @param graph_shapes A list of shapes for each graph argument.
        * @param graph_types A list of types for each graph argument.
        * @return Returns the compiled XLA computation with refined shapes.
        */
        void compile(
            const int n_atoms,
            std::vector<std::vector<int64_t>> graph_shapes,
            std::vector<xla::PrimitiveType> graph_types
        );

        mlir::ModuleOp module() const { return module_ref.get(); }

    private:
	    mlir::MLIRContext context;
        mlir::MLIRContext export_context;

        std::string mlir_module;

        mlir::OwningOpRef<mlir::ModuleOp> module_ref;
    };
}






#endif //COMPILE_H
