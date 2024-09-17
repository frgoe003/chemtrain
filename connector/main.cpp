#include "main.h"
#include <iostream>
#include <string>
#include <vector>
#include <memory>  // For std::unique_ptr
#include <dlfcn.h>

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/service/dump.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

#include "compile.h"

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
        Impl(const int max_neighbors, std::string hlo_filename) : max_neighbors(max_neighbors), compiler(hlo_filename) {

          	std::cout << "Try to load GPU Plugin" << std::endl;

//            backend = dlopen("/venv/lib/python3.11/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so", RTLD_LAZY);
//            if (!backend) {
//                std::cerr << "Cannot open library: " << dlerror() << std::endl;
//                return;
//            }
//
//            *reinterpret_cast<void**>(&init_client_fn) = dlsym(library, "GetStreamExecutorGpuClient");
//			*reinterpret_cast<void**>(&init_options_fn) = dlsym(library, "CreatePjRtClientOptions");
//
//            absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_or_status = init_client_fn(init_options_fn());
//
//            if (!client_or_status.ok()) {
//                std::cerr << "Cannot create client: " << client_or_status.status().ToString() << std::endl;
//                client_ = xla::GetTfrtCpuClient(/*asynchronous=*/false).value();
//            } else {
//             	client_ = std::move(client_or_status).value();
//            }

            // Load the CUDA plugin from JAX
            absl::StatusOr<const PJRT_Api*> status_or_api = pjrt::LoadPjrtPlugin(
                    "CUDA",
                    "/venv/lib/python3.11/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so");

            if (!status_or_api.ok()) {
            	LOG(INFO) << "Failed to load CUDA plugin: " << status_or_api.status();
                return;
            }

            if (!pjrt::IsPjrtPluginInitialized("CUDA").value()) {
                std::cerr << "Initialize CUDA plugin" << std::endl;

                pjrt::InitializePjrtPlugin("CUDA");
            }

            absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_or_status = xla::GetCApiClient("CUDA");
            if (!client_or_status.ok()) {
                std::cerr << "Cannot create client: " << client_or_status.status().ToString() << std::endl;
                client_ = xla::GetTfrtCpuClient(/*asynchronous=*/true).value();
            } else {
                client_ = std::move(client_or_status).value();
            }

            // Create client to do buffer transfers
            host_ = xla::GetTfrtCpuClient(/*asynchronous=*/false).value();

            std::cout << "Found addressable devices: ";
            for (int i = 0; i < client_->addressable_devices().size(); i++) {
                std::cout << client_->addressable_devices()[i] << " ";
            }
            std::cout << std::endl;

//            // Initialization code related to XLA
//            std::string hlo_string;
//            tsl::ReadFileToString(tsl::Env::Default(), hlo_filename, &hlo_string);
//            hlo_string = StripLogHeaders(hlo_string);
//
//
//            // For HLO strings
//            // std::unique_ptr<xla::HloModule> test_module = ParseAndReturnUnverifiedModule(hlo_string, xla::HloModuleConfig()).value();
//            // const xla::HloModuleProto test_module_proto = test_module->ToProto();
//
//            // Load .pb file:
//            xla::HloSnapshot proto;
//            if (!proto.ParseFromString(hlo_string) &&
//                !proto.mutable_hlo()->ParseFromString(hlo_string) &&
//                !proto.mutable_hlo()->mutable_hlo_module()->ParseFromString(hlo_string)) {
//            	std::cout << "Failed to parse input as HLO protobuf binary" << std::endl;
//            }
//
//            // Note: We always call .value() since XLA always returns a status wrapper.
//            xla::DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
//            xla::HloModuleConfig config = xla::HloModule::CreateModuleConfigFromProto(proto.hlo().hlo_module(), debug_options).value();
//            std::unique_ptr<xla::HloModule> test_module = xla::HloModule::CreateFromProto(proto.hlo().hlo_module(), config).value();

            // New: Automatically compile

            const int n_atoms = 10;

            std::cout << "Try to compile the file" << std::endl;
            mlir::MLIRContext context;
            // mlir::OwningOpRef<mlir::ModuleOp> module = compiler.compile(n_atoms, max_neighbors, context);
            xla::XlaComputation computation = compiler.compile(n_atoms, max_neighbors, context);

            // module->dump();

            std::cout << "Compile the module" << std::endl;

            xla::CompileOptions compile_options;
            // auto executable_or_status = client_->Compile(*module, compile_options);
            auto executable_or_status = client_->Compile(computation, compile_options);
            std::cout << "Client creation status: " << executable_or_status.status().ToString() << std::endl;
            executable_ = std::move(executable_or_status).value();
            std::cout << "Executable created" << std::endl;
        }

        std::vector<std::vector<float>> execute(const std::vector<std::vector<float>>& position, const std::vector<std::vector<int>>& neighbors) {

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

            // We have to create a buffer for the input data.
            // Create the buffer on the host and transfer to the devices
            // std::cout << "Buffer creation..." << std::endl;
            xla::Literal *literal_pointer;

            std::unique_ptr<xla::PjRtBuffer> param_x = client_->BufferFromHostBuffer(
            	literal_x.untyped_data(),
        		literal_x.shape().element_type(),
        		literal_x.shape().dimensions(), std::optional<absl::Span<int64_t const>>{},
        		xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
        		[]() { /* frees literal */ }, client_->addressable_devices()[0]).value();

            std::unique_ptr<xla::PjRtBuffer> param_y = client_->BufferFromHostBuffer(
            	literal_y.untyped_data(),
        		literal_y.shape().element_type(),
        		literal_y.shape().dimensions(), std::optional<absl::Span<int64_t const>>{},
        		xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
        		[]() { /* frees literal */ }, client_->addressable_devices()[0]).value();


            // # std::unique_ptr<xla::PjRtBuffer> param_y = host_->BufferFromHostLiteral(literal_y, host_->addressable_devices()[0]).value();
            // std::unique_ptr<xla::PjRtBuffer> param_x_host = host_->BufferFromHostLiteral(literal_x, host_->addressable_devices()[0]);
            // std::unique_ptr<xla::PjRtBuffer> param_y_host = host_->BufferFromHostLiteral(literal_y, host_->addressable_devices()[0]).value();


            // std::cout << "Execute..." << std::endl;
            xla::ExecuteOptions execute_options;
            std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results = executable_->Execute({{param_x.get(), param_y.get()}}, execute_options).value();

            // std::cout << "Finished execution" << std::endl;

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

        const int max_neighbors;

        void * backend;

        std::unique_ptr<xla::PjRtLoadedExecutable> executable_;
        std::unique_ptr<xla::PjRtClient> client_;
       	std::unique_ptr<xla::PjRtClient> host_;

        Compiler compiler;

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

    Connector::Connector(const int max_neighbors, const std::string hlo_path) : impl_(std::make_unique<Impl>(max_neighbors, hlo_path)) {}
    Connector::~Connector() = default;

    std::vector<std::vector<float>> Connector::force(const std::vector<std::vector<float>>& position, const std::vector<std::vector<int>>& neighbors) {
        return impl_->execute(position, neighbors);
    }

    void execute() {

        const int max_neighbors = 5;
        Connector connector = Connector(max_neighbors, "./fn_hlo.txt");

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

int main(int argc, char** argv){
    jcn::execute();
    return 0;
}
