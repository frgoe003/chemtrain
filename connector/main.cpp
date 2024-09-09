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

#include "main.h"

#include "xla/literal.h"
#include "xla/literal_util.h"

#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"


#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
// #include "xla/pjrt/gpu/se_gpu_pjrt_client.h"

#include "xla/status.h"
#include "xla/statusor.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

template <typename T>
void print2DVector(const std::vector<std::vector<T>>& vec) {
    for (const auto& row : vec) {
        for (const auto& elem : row) {
            std::cout << elem << " ";  // Print each element followed by a space
        }
        std::cout << std::endl;  // Print a newline at the end of each row
    }
}


std::string jcn::StripLogHeaders(std::string_view hlo_string);


class jcn::Connector::Connect {

    public:
      std::unique_ptr<xla::PjRtLoadedExecutable> executable;
      std::unique_ptr<xla::PjRtClient> client;

      Connect () {

            // Load HloModule from file.
            std::string hlo_filename = "./fn_hlo.txt";

            std::string hlo_string;
            tsl::ReadFileToString(tsl::Env::Default(), hlo_filename, &hlo_string);
            hlo_string = jcn::StripLogHeaders(hlo_string);


            std::unique_ptr<xla::HloModule> test_module = ParseAndReturnUnverifiedModule(hlo_string, xla::HloModuleConfig()).value();
            const xla::HloModuleProto test_module_proto = test_module->ToProto();

            // Run it using JAX C++ Runtime (PJRT).


            // Get a CPU client.
            client = xla::GetTfrtCpuClient(/*asynchronous=*/false).value();

            // Compile XlaComputation to PjRtExecutable.
            xla::XlaComputation xla_computation(test_module_proto);
            xla::CompileOptions compile_options;

            // We initialized the module
            auto executable_or_status =
                client->Compile(xla_computation, compile_options);

            std::cout << "Client creation status: " << executable_or_status.status().ToString() << std::endl;

            executable = std::move(executable_or_status).value();

            std::cout << "Executable created" << std::endl;
        }

};


jcn::Connector::Connector(): connect_instance(std::make_unique<Connect>()) {}


std::vector<std::vector<float>> jcn::Connector::force(std::vector<std::vector<float>> position, std::vector<std::vector<int>> neighbors) {

    // Could be set to a better value
    // int max_neighbors = 9;
    int max_neighbors = 4;


    // Create arrays in the correct format. Directly fill in the neighbor list
    // with invalid values.
    xla::Array2D<float> position_array = xla::Array2D<float>(position.size(), 3);
    xla::Array2D<int> neighbor_array = xla::Array2D<int>(neighbors.size(), max_neighbors, neighbors.size());

    // Fill in the data of the arrays
    for (int i = 0; i < position.size(); i++) {
         // Fill in all positions
        for (int j = 0; j < 3; j++) {
            position_array(i, j) = position[i][j];
        }

        // Fill in all neighbors
        for (int j = 0; j < neighbors[i].size(); j++) {
            neighbor_array(i, j) = neighbors[i][j];
        }

    }

    // Create the literal from the Array2D
    xla::Literal literal_x = xla::LiteralUtil::CreateFromArray(position_array);
    xla::Literal literal_y = xla::LiteralUtil::CreateFromArray(neighbor_array);

    auto buffer_or_status = connect_instance->client->BufferFromHostLiteral(literal_x, connect_instance->client->addressable_devices()[0]);

    std::cout << "Buffer creation status: " << buffer_or_status.status().ToString() << std::endl;

    std::unique_ptr<xla::PjRtBuffer> param_x = std::move(buffer_or_status).value();
    std::unique_ptr<xla::PjRtBuffer> param_y =
        connect_instance->client->BufferFromHostLiteral(literal_y, connect_instance->client->addressable_devices()[0])
          .value();

    std::cout << "Execute..." << std::endl;

    // Execute on CPU.
    xla::ExecuteOptions execute_options;
    // One vector<buffer> for each device.
    std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results =
        connect_instance->executable->Execute({{param_x.get(), param_y.get()}}, execute_options)
            .value();

    // Get result.
    std::shared_ptr<xla::Literal> result_literal =
        results[0][0]->ToLiteralSync().value();
    auto flat_results = result_literal->data<float>();

    // Copy result into vector
    std::vector<std::vector<float>> result;
    for (int i = 0; i < position.size(); i++) {
        std::vector<float> new_col;

        for (int j = 0; j < 3; j++) {
            new_col.push_back(flat_results[i * 3 + j]);
        }
        result.push_back(new_col);
    }

    return result;
}


std::string jcn::StripLogHeaders(std::string_view hlo_string) {
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


int main(int argc, char** argv) {
  tsl::port::InitMain("", &argc, &argv);

  jcn::execute();

    return 0;
}


void jcn::execute() {

  jcn::Connector connector;

  int atoms = 5;

  std::vector<std::vector<float>> position = {};
  for (float x = 0.f; x<atoms; x++){
	std::vector<float> new_col = {0.1f * x, 0.1f * x + 1.0f, 0.1f * x - 1.0f};
	position.push_back(new_col);
  }

  std::vector<std::vector<int>> neighbors = {};
    for (int i = 0; i < position.size(); i++) {
        std::vector<int> new_col;
	    for (int j = 0 ; j < position.size(); j++) {
		    if (i > j) {
		        new_col.push_back(j);
		    }
		    if (i < j) {
			    new_col.push_back(j);
		    }
	    }
         neighbors.push_back(new_col);
    }

    std::cout << "Neighbors: " << std::endl;
    print2DVector<int>(neighbors);

    std::vector<std::vector<float>> result = connector.force(position, neighbors);

    std::cout << "Result: " << std::endl;
    print2DVector<float>(result);

    position = {};
    for (float x = 0.f; x<atoms; x++){
        std::vector<float> new_col = {0.2f * x, 0.1f * x + 1.0f, 0.1f * x - 1.0f};
        position.push_back(new_col);
    }

    result = connector.force(position, neighbors);

    std::cout << "Result: " << std::endl;
    print2DVector<float>(result);

}
