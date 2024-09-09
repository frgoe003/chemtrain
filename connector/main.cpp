#include "main.h"
#include <iostream>
#include <string>
#include <vector>
#include <memory>  // For std::unique_ptr

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
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
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
}

// Implementation of the Connector::Impl class
namespace jcn {
    class Connector::Impl {
    public:
        Impl() {
            // Initialization code related to XLA
            std::string hlo_filename = "./fn_hlo.txt";
            std::string hlo_string;
            tsl::ReadFileToString(tsl::Env::Default(), hlo_filename, &hlo_string);
            hlo_string = StripLogHeaders(hlo_string);

            std::unique_ptr<xla::HloModule> test_module = ParseAndReturnUnverifiedModule(hlo_string, xla::HloModuleConfig()).value();
            const xla::HloModuleProto test_module_proto = test_module->ToProto();

            client_ = xla::GetTfrtCpuClient(/*asynchronous=*/false).value();
            xla::XlaComputation xla_computation(test_module_proto);
            xla::CompileOptions compile_options;

            auto executable_or_status = client_->Compile(xla_computation, compile_options);
            std::cout << "Client creation status: " << executable_or_status.status().ToString() << std::endl;
            executable_ = std::move(executable_or_status).value();
            std::cout << "Executable created" << std::endl;
        }

        std::vector<std::vector<float>> execute(const std::vector<std::vector<float>>& position, const std::vector<std::vector<int>>& neighbors) {
            int max_neighbors = 4;

            xla::Array2D<float> position_array(position.size(), 3);
            xla::Array2D<int> neighbor_array(neighbors.size(), max_neighbors, neighbors.size());

            for (int i = 0; i < position.size(); i++) {
                for (int j = 0; j < 3; j++) {
                    position_array(i, j) = position[i][j];
                }

                for (int j = 0; j < neighbors[i].size(); j++) {
                    neighbor_array(i, j) = neighbors[i][j];
                }
            }

            xla::Literal literal_x = xla::LiteralUtil::CreateFromArray(position_array);
            xla::Literal literal_y = xla::LiteralUtil::CreateFromArray(neighbor_array);

            auto buffer_or_status = client_->BufferFromHostLiteral(literal_x, client_->addressable_devices()[0]);
            std::cout << "Buffer creation status: " << buffer_or_status.status().ToString() << std::endl;

            std::unique_ptr<xla::PjRtBuffer> param_x = std::move(buffer_or_status).value();
            std::unique_ptr<xla::PjRtBuffer> param_y = client_->BufferFromHostLiteral(literal_y, client_->addressable_devices()[0]).value();

            std::cout << "Execute..." << std::endl;
            xla::ExecuteOptions execute_options;
            std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results = executable_->Execute({{param_x.get(), param_y.get()}}, execute_options).value();

            std::shared_ptr<xla::Literal> result_literal = results[0][0]->ToLiteralSync().value();
            auto flat_results = result_literal->data<float>();

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

    private:
        std::unique_ptr<xla::PjRtLoadedExecutable> executable_;
        std::unique_ptr<xla::PjRtClient> client_;

        std::string StripLogHeaders(std::string_view hlo_string) {
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
    };

    Connector::Connector() : impl_(std::make_unique<Impl>()) {}
    Connector::~Connector() = default;

    std::vector<std::vector<float>> Connector::force(const std::vector<std::vector<float>>& position, const std::vector<std::vector<int>>& neighbors) {
        return impl_->execute(position, neighbors);
    }

    void execute() {
        Connector connector;

        int atoms = 5;
        std::vector<std::vector<float>> position;
        for (float x = 0.f; x < atoms; x++) {
            std::vector<float> new_col = {0.1f * x, 0.1f * x + 1.0f, 0.1f * x - 1.0f};
            position.push_back(new_col);
        }

        std::vector<std::vector<int>> neighbors;
        for (int i = 0; i < position.size(); i++) {
            std::vector<int> new_col;
            for (int j = 0; j < position.size(); j++) {
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

        position.clear();
        for (float x = 0.f; x < atoms; x++) {
            std::vector<float> new_col = {0.2f * x, 0.1f * x + 1.0f, 0.1f * x - 1.0f};
            position.push_back(new_col);
        }

        result = connector.force(position, neighbors);

        std::cout << "Result: " << std::endl;
        print2DVector<float>(result);
    }
}
