/* Copyright 2021 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example for reading a HloModule from a HloProto file and execute the
// module on PJRT CPU client.
//
// To build a HloModule,
//
// $ python3 jax/tools/jax_to_hlo.py \
// --fn examples.jax_cpp.prog.fn \
// --input_shapes '[("x", "f32[2,2]"), ("y", "f32[2,2]")]' \
// --constants '{"z": 2.0}' \
// --hlo_text_dest /tmp/fn_hlo.txt \
// --hlo_proto_dest /tmp/fn_hlo.pb
//
// To load and run the HloModule,
//
// $ bazel build examples/jax_cpp:main --experimental_repo_remote_exec --check_visibility=false
// $ bazel-bin/examples/jax_cpp/main
// 2021-01-12 15:35:28.316880: I examples/jax_cpp/main.cc:65] result = (
// f32[2,2] {
//   { 1.5, 1.5 },
//   { 3.5, 3.5 }
// }
// )

#include <memory>
#include <string>
#include <vector>

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"


std::string StripLogHeaders(std::string_view hlo_string) {
  // I0521 12:04:45.883483    1509 service.cc:186] ...
  static RE2* matcher = new RE2(
      "[IWEF]\\d{4} "
      "\\d{2}:\\d{2}:\\d{2}\\.\\d+\\s+\\d+\\s+[^:]+:\\d+\\]\\s?(.*)");
  std::string_view matches[4];
  std::vector<std::string> lines = absl::StrSplit(hlo_string, '\n');
  for (auto& line : lines) {
    if (matcher->Match(line, 0, line.size(), RE2::ANCHOR_START, matches, 4)) {
      line = std::string(matches[1]);
    }
  }
  return absl::StrJoin(lines, "\n",
                       [](std::string* out, const std::string& line) {
                         absl::StrAppend(out, line);
                       });
}

void execute(int atoms);

int main(int argc, char** argv) {
  tsl::port::InitMain("", &argc, &argv);

  const int atoms = int(*argv[1] - '0');
  execute(atoms);

    return 0;
}


void execute(int atoms) {

  // Load HloModule from file.
  std::string hlo_filename = "./fn_hlo.txt";
//  std::function<void(xla::HloModuleConfig*)> config_modifier_hook =
//      [](xla::HloModuleConfig* config) { config->set_seed(42); };

  std::cout << "Read in file" << std::endl;

  std::string hlo_string;
  tsl::ReadFileToString(tsl::Env::Default(), hlo_filename, &hlo_string);

  std::cout << "File is: " << hlo_string << std::endl;

  hlo_string = StripLogHeaders(hlo_string);

  std::cout << "Stripped log headers: " << hlo_string << std::endl;

   std::unique_ptr<xla::HloModule> test_module = ParseAndReturnUnverifiedModule(hlo_string, xla::HloModuleConfig()).value();
   const xla::HloModuleProto test_module_proto = test_module->ToProto();

  // Run it using JAX C++ Runtime (PJRT).

   std::cout << "Starting execution on client" << std::endl;

  // Get a CPU client.
  std::unique_ptr<xla::PjRtClient> client =
      xla::GetTfrtCpuClient(/*asynchronous=*/true).value();

  // Compile XlaComputation to PjRtExecutable.
  xla::XlaComputation xla_computation(test_module_proto);
  xla::CompileOptions compile_options;
  std::unique_ptr<xla::PjRtLoadedExecutable> executable =
      client->Compile(xla_computation, compile_options).value();

  std::vector<std::vector<float>> data = {};
  for (float x = 0.f; x<atoms; x++){
	std::vector<float> new_col = {0.1f * x, 0.1f * x + 1.0f, 0.1f * x - 1.0f};
	data.push_back(new_col);
  }
  // xla::Shape x_shape = xla::ShapeUtil::MakeShape(xla::F32, absl::Span(std::vector<int>(x1, x2)));

  // Prepare inputs.

    // Define the data
    //std::vector<std::vector<float>> data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};

    // Create an Array2D from the data
    xla::Array<float> array2d = xla::Array2D<float>(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[i].size(); ++j) {
            array2d(i, j) = data[i][j];
        }
    }

    xla::Array<int> nbrs = xla::Array2D<int>(data.size(), data.size() - 1);
    for (int i = 0; i < data.size(); i++) {
	    for (int j = 0 ; j < data.size(); j++) {
		    if (i > j) {
		            nbrs(i, j) = j;
		    }
		    if (i < j) {
			    nbrs(i, j - 1) = j;
		    }
	    }
    }

    // Create the literal from the Array2D
    xla::Literal literal_x = xla::LiteralUtil::CreateFromArray(array2d);
    xla::Literal literal_y = xla::LiteralUtil::CreateFromArray(nbrs);

    // Print the literal to verify
    std::cout << literal_x.ToString() << std::endl;
    std::cout << literal_y.ToString() << std::endl;


  std::unique_ptr<xla::PjRtBuffer> param_x =
      client->BufferFromHostLiteral(literal_x, client->addressable_devices()[0])
          .value();
  std::unique_ptr<xla::PjRtBuffer> param_y =
      client->BufferFromHostLiteral(literal_y, client->addressable_devices()[0])
          .value();

  // Execute on CPU.
  xla::ExecuteOptions execute_options;
  // One vector<buffer> for each device.
  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results =
      executable->Execute({{param_x.get(), param_y.get()}}, execute_options)
          .value();

  // Get result.
  std::shared_ptr<xla::Literal> result_literal =
      results[0][0]->ToLiteralSync().value();
  LOG(INFO) << "result = " << *result_literal;
}
