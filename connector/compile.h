//
// Created by Paul Fuchs on 13.09.24.
//

#include "pybind11/embed.h"

#include <string>
#include "xla/service/hlo_parser.h"

#ifndef COMPILE_H
#define COMPILE_H

namespace py = pybind11;

namespace jcn {
    class Compiler {
    public:
        Compiler(std::string py_executable);
        ~Compiler() = default;

        // Takes the python file and compiles the force_fn givent the number of
        // particles and the maxmimum number of neighbors
        absl::StatusOr<std::unique_ptr<xla::HloModule>> compile(
            const int n_atoms, const int max_neighbor);

    private:
        py::object force_compiler;

        // Keeps the interpreter alive as long as the class exists.
        py::scoped_interpreter guard{};

    };
}






#endif //COMPILE_H
