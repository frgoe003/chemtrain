# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//jaxlib:jax.bzl",
    "py_library_providing_imports_info",
    "pybind_extension",
    "pytype_library"
)
load("@rules_python//python:defs.bzl", "py_binary")

py_runtime(
    name = "my_py_runtime",
    interpreter_path = "/usr/bin/python3",
    python_version = "PY3",
)

cc_binary(
    name = "libmain.so",
    srcs = [
        "main.cpp",
        "main.h",
        ],
    tags = ["manual"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:status",
        "@xla//xla:statusor",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_stream_executor_client",
        "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service:hlo_parser",
        "@xla//xla/service:dump",
        ":compile"

        # "@xla//xla/tools:hlo_module_loader",
    ],
    # linkopts=['-static'],
    linkopts = [
        "-Wl,--no-undefined",
    ],
    linkshared=True
)

alias(
    name = "mlir_to_hlo_alias",
    actual = "@xla//xla/pjrt:mlir_to_hlo",
    visibility = ["//visibility:public"],  # This alias is publicly accessible
)


cc_library(
    name = "compile",
    srcs = ["compile.cpp",],
    hdrs = ["compile.h"],
    deps = [
        ":xla_call_module_loader",
        "@xla//xla/client:xla_computation",
        "@xla//xla/service:hlo_parser",
        "@xla//xla/mlir_hlo:stablehlo_extension_passes",
        ":mlir_to_hlo_alias",
        "@xla//third_party/python_runtime:headers",
        "@xla//xla/mlir_hlo:hlo_dialect_registration",
        "@xla//xla/mlir_hlo:mhlo_passes",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:logging",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        # "@llvm-project//mlir:Function",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLProgramDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@shardy//shardy/dialect/sdy/ir:register",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:version",
        "@pybind11",
    ],
    linkopts = ["-lpython3.11"],  # Adjust the version as necessary
)

cc_library(
    name = "xla_call_module_loader",
    srcs = ["xla_call_module_loader.cpp"],
    hdrs = ["xla_call_module_loader.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
        "@xla//xla:shape_util",
        "@xla//xla/client:xla_computation",
        "@xla//xla/mlir/utils:type_util",
        "@xla//xla/mlir_hlo",
        "@xla//xla/mlir_hlo:mhlo_passes",
        "@xla//xla/python:refine_polymorphic_shapes",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:vhlo_ops",
    ],
)

