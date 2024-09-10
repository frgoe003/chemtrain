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

cc_binary(
    name = "libmain.so",
    srcs = ["main.cpp", "main.h"],
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
        # "@xla//xla/tools:hlo_module_loader",
    ],
    # linkopts=['-static'],
    linkopts = [
        "-Wl,--no-undefined",
    ],
    linkshared=True
)

cc_binary(
    name = "pjrt_c_api_gpu_plugin.so",
    linkopts = [
        "-Wl,--version-script,$(location @xla//xla/pjrt/c:pjrt_c_api_gpu_version_script.lds)",
        "-Wl,--no-undefined",
    ],
    tags = [
        "noasan",
        "nomac",  # No GPU on mac.
        "nomsan",
        "notsan",
    ],
    linkshared = True,
    deps = [
        "@xla//xla/pjrt/c:pjrt_c_api_gpu",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu_version_script.lds",
        "@xla//xla/service:gpu_plugin",
        "@xla//xla/stream_executor:cuda_platform",
    ]
)


cc_library(
    name = "connector",
    srcs = ["main.cpp"],
    tags = ["manual"],
    hdrs = ["main.h"],
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
        # "@xla//xla/tools:hlo_module_loader",
    ],
)


# load("@xla//xla:xla_package_groups.bzl", "xla_packages")

# visibility(xla_packages)


