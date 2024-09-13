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

cc_library(
    name = "compile",
    srcs = ["compile.cpp",],
    hdrs = ["compile.h"],
    deps = [
        "@xla//xla/client:xla_computation",
        "@xla//xla/service:hlo_parser",
        "@xla//third_party/python_runtime:headers",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:logging",
        "@pybind11",
    ],
    linkopts = ["-lpython3.11"],  # Adjust the version as necessary
)

