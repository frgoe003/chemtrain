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

#include "tsl/platform/protobuf.h"
#include "tsl/platform/logging.h"


namespace py = pybind11;

namespace jcn {

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
            from jax.core import ShapedArray

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

                    # Import the force function
                    self.force_fn = jax.jit(module.force_fn)

                    print('Finished initializing the force function')

                def compile(self, n_atoms, max_neighbors):
                    # To avoid specifying data, we use only a ShapedArray for
                    # lowering

                    position = ShapedArray((n_atoms, 3), jnp.float32)
                    neighbor = ShapedArray((n_atoms, max_neighbors), jnp.int32)

                    print('Start compiling the force function')

                    return self.force_fn.lower(
                        position, neighbor
                        ).compiler_ir('hlo').as_serialized_hlo_module_proto()
            )");

            std::cout << "Get the class" << std::endl;

            py::object ForceModule = py::globals()["ForceModule"];

            std::cout << "Instanciate it" << std::endl;

            // We import the lowering class using pybind
            force_compiler = ForceModule(py_executable);


        } catch (const py::error_already_set& e) {
            std::cerr << "Python error: " << e.what() << std::endl;
        }

        std::cout << "Finished initialization" << std::endl;
    }

    absl::StatusOr<std::unique_ptr<xla::HloModule>> Compiler::compile(
        const int n_atoms, const int max_neighbor) {
        xla::HloSnapshot proto;
        std::unique_ptr<xla::HloModule> module;


        std::string bytestring;
        try {
            // We import the decorated module and get the serialized HLO module as a
            // bytestring
            py::gil_scoped_acquire acquire;

            std::cout << "Start compiling the force function" << std::endl;
            bytestring = force_compiler.attr(
                "compile")(n_atoms, max_neighbor).cast<std::string>();

        } catch (const py::error_already_set& e) {
            std::cerr << "Python error: " << e.what() << std::endl;
        }

        // We try to serialize the protobuffer
        if (!proto.ParseFromString(bytestring) &&
            !proto.mutable_hlo()->ParseFromString(bytestring) &&
            !proto.mutable_hlo()->mutable_hlo_module()->ParseFromString(bytestring)) {
            return tsl::errors::InvalidArgument("Failed to parse input as HLO protobuf binary");
        }

        xla::DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
        TF_ASSIGN_OR_RETURN(
            xla::HloModuleConfig config,
            xla::HloModule::CreateModuleConfigFromProto(
                proto.hlo().hlo_module(), debug_options));
        TF_ASSIGN_OR_RETURN(
            module, xla::HloModule::CreateFromProto(proto.hlo().hlo_module(), config));

        return std::move(module);
    }


} // namespace jcn
