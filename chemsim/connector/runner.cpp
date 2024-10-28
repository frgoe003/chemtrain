//
// Created by Paul Fuchs on 24.09.24.
//

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>  // For std::unique_ptr
#include <dlfcn.h>
#include <filesystem>
#include <regex>

#include "connector/runner.h"
#include "connector/compiler.h"
#include "connector/libconnector.h"
#include "connector/domain.h"
#include "connector/pjrt.h"
#include "connector/model.pb.h"
#include "connector/utils.h"

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/dump.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"


namespace jcn {

    void initialize() {

        // TODO: Make this more general
        std::cout << "Try to load GPU Plugin" << std::endl;

        char* env_var = std::getenv("JCN_PJRT_PATH");
        if (env_var == nullptr) {
            std::cerr << "Set JCN_PJRT_PATH envvar to discover PJRT Plugins" << std::endl;
            return;
        }

        std::string plugin_path = std::string(std::getenv("JCN_PJRT_PATH"));


        try {
            // Infer a name
            std::regex pattern(R"(jax_plugins\.xla_(\w+)\.so)");

            for (const auto& entry : std::filesystem::directory_iterator(plugin_path)) {
                std::string path = entry.path().string();
                std::smatch match;

                if (!std::regex_search(path, match, pattern)) continue;

                absl::StatusOr<const PJRT_Api*> status_or_api = pjrt::LoadPjrtPlugin(match.str(1), path);

                if (status_or_api.ok()) {
                    std::cout << "Loaded plugin: " << match.str(1) << std::endl;
                }

            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to load pjrt plugins: " << e.what() << std::endl;
        }

    }


    Runner::Runner(ConnectorConfig config) :
        atom_builder(std::make_unique<AtomBuilder>(config.atom_multiplier)),
        model(std::make_unique<chemsim::Model>())
    {

        // Initialize PJRT
        initialize();

        // Singleton
        Logger logger = Logger::getlogger();

        compile_options.profile_version = 9;

        compile_options.executable_build_options = compile_options.executable_build_options.set_run_backend_only(false);
        compile_options.executable_build_options = compile_options.executable_build_options.set_device_ordinal(config.device);
        compile_options.executable_build_options = compile_options.executable_build_options.set_deduplicate_hlo(true);

        logger.log(LogLevel::DEBUG, "Build options: " + compile_options.executable_build_options.ToString());

        // Deserialize the protobuffer
        model->ParseFromString(config.model);

        // Pass the mlir module to the compiler
        compiler = std::make_unique<Compiler>(model->mlir_module());

        // Select from the available neighbor list types
        switch (model->neighbor_list().type()) {
            case chemsim::Model::SIMPLE_SPARSE:
                neighbor_list = std::make_unique<SimpleSparseNeighborList>();
                neighbor_list->initialize(config.neighbor_list_multipliers);

                logger.log(LogLevel::INFO, "Initialize SimpleSparseNeighborList");
                break;
            case chemsim::Model::DEVICE_SPARSE:
                neighbor_list = std::make_unique<DeviceSparseNeighborList>();
                neighbor_list->initialize(config.neighbor_list_multipliers);

                logger.log(LogLevel::INFO, "Initialize DeviceSparseNeighborList");
                break;
        }

        // Initialize the possible backends in the libconnector file
        absl::StatusOr<bool> status_or_success = pjrt::IsPjrtPluginInitialized(config.backend);
        if (!status_or_success.ok()) {
            throw std::runtime_error("Failed to initialize PjRtClient: " + status_or_success.status().ToString());
        }

        if (!status_or_success.value()) {
            absl::Status status = pjrt::InitializePjrtPlugin(config.backend);
            if (!status.ok()) {
                throw std::runtime_error("Failed to initialize PjRtClient: " + status.ToString());
            }
        }

        absl::flat_hash_map<std::string, xla::PjRtValueType> create_options = {
            {"memory_fraction", xla::PjRtValueType(0.95f)},
        };

        // Get the client
        absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_or_status = xla::GetCApiClient(config.backend, create_options);
        if (!client_or_status.ok()) {
            client = xla::GetTfrtCpuClient(/*asynchronous=*/true).value();
        } else {
            client = std::move(client_or_status).value();
        }

    }

    double Runner::compute_forces(
        int inum, int gnum, double **x, double **f, int *type, int *ilist,
        int *numneigh, int **firstneigh, bool list_changed
    ) {

        // Singleton
        Logger logger = Logger::getlogger();

        int max_trials = 10;

        for (int i = 0; i < max_trials; i++) {

            auto trial_start = std::chrono::high_resolution_clock::now();

            // First we build the domain and the neighbor list, then we can
            // determine the input shapes to the program

            AtomShapes atoms = atom_builder->get_shapes(inum, gnum);
            NeighborListShapes neighbors = neighbor_list->get_neighbor_list_shapes(
                atoms.n_atoms, inum, numneigh);

            // Now we have all shapes setup to build the module if required
            if (!executable || atoms.reallocate || neighbors.reallocate ) {
                std::cout << "Recompilation necessary" << std::endl;

                compiler->compile(
                    atoms.n_atoms, neighbors.graph_shapes, neighbors.graph_types);

                absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> executable_or_status = client->Compile(
                    compiler->module(), compile_options);

                if (!executable_or_status.ok()) {
                    throw std::runtime_error("Failed to compile: " + executable_or_status.status().ToString());
                }

                executable = std::move(executable_or_status).value();
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Only transfer new data to the GPU if necessary
            bool update = (neighbors.reallocate || list_changed);

            // Now we have to create the buffers, i.e., copy the data onto
            // the device
            std::vector<xla::PjRtBuffer*> buffer_ptrs = atom_builder->build_domain(client.get(), config.device, inum, gnum, x, type);

            std::vector<xla::PjRtBuffer*> graph_buffers = neighbor_list->build_graph(
                client.get(), config.device, inum, ilist, numneigh, firstneigh, update);
            buffer_ptrs.insert(buffer_ptrs.end(), graph_buffers.begin(), graph_buffers.end());

            std::vector<std::vector<xla::PjRtBuffer*>> arg_handles = {buffer_ptrs};

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            logger.log(LogLevel::DEBUG, "Time taken for buffer creation: " + std::to_string(duration.count()) + " seconds");

            // Check if arg_handles is correctly populated
            if (arg_handles.empty() || arg_handles[0].empty()) {
                throw std::runtime_error("arg_handles is empty or not properly populated");
            }

            // No idea what to specify here...
            xla::ExecuteOptions execute_options;

            start = std::chrono::high_resolution_clock::now();

            absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>> results;
            results = executable->Execute(
                absl::Span<const std::vector<xla::PjRtBuffer*>>(arg_handles),
                execute_options
            );

            if (!results.ok()) {
                throw std::runtime_error("Failed to execute: " + results.status().ToString());
            }

            // Now we have to copy the results back to the host
            std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results_buffers = std::move(results).value();

            bool success = neighbor_list->evaluate_statistics(results_buffers);

            end = std::chrono::high_resolution_clock::now();
            duration = end - start;

            logger.log(LogLevel::DEBUG, "Time taken for computation: " + std::to_string(duration.count()) + " seconds");

            // Write back the results
            double potential = atom_builder->evaluate_domain(
                success, inum, f, results_buffers);

            auto trial_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> trial_duration = trial_end - trial_start;

            logger.log(LogLevel::DEBUG, "Time taken for trial: " + std::to_string(trial_duration.count()) + " seconds");

            results_buffers.clear();

            // Finished
            if (success) return potential;

        }

        throw std::runtime_error("Failed to compute forces after " + std::to_string(max_trials) + " trials");

    }

    ModelProperties Runner::get_model_properties() {
        ModelProperties properties;

        switch (model->neighbor_list().type()) {
            case chemsim::Model::SIMPLE_SPARSE:
                // Neighbor list cutoff must be larger than the model cutoff
                properties.cutoff = model->neighbor_list().cutoff();

                if (model->neighbor_list().has_num_mpl()) {
                    // Ghost atoms sending edges only required if we perform message passing
                    properties.neighbor_list.include_ghosts = (model->neighbor_list().num_mpl() > 0);
                    std::cout << "Include ghosts: " << properties.neighbor_list.include_ghosts << std::endl;
                };
                if (model->neighbor_list().has_half_list()) {
                    properties.neighbor_list.half_list = model->neighbor_list().half_list();
                    std::cout << "Use half list only " << properties.neighbor_list.half_list << std::endl;
                };

                break;
            case chemsim::Model::DEVICE_SPARSE:
                // Does not specify a cutoff for the particles as neighbor
                // list is computed on the device
                properties.cutoff = 0.0;

                // Atoms still need to be communicated
                properties.comm_dist = model->neighbor_list().cutoff();
                if (model->neighbor_list().has_num_mpl()) {
                    properties.comm_dist *= model->neighbor_list().num_mpl();
                }
                break;
        }

        return properties;
    }

} // namespace jcn
