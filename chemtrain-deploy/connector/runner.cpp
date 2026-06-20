/*
Copyright 2025 Multiscale Modeling of Fluid Materials, TU Munich

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

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <regex>
#include <future>
#include <cstdlib>

#include "connector/runner.h"
#include "connector/compiler.h"
#include "connector/libconnector.h"
#include "connector/domain.h"
#include "connector/buffer.h"
#include "connector/model.pb.h"
#include "connector/utils.h"
#include "connector/openequivariance.h"
#include "connector/communication.h"

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/log/initialize.h"
#include "xla/service/dump.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"


namespace jcn {

    void Runner::initialize() {

        absl::InitializeLog();

        Logger logger = Logger::getlogger();


        const char* raw_env = std::getenv("JCN_PJRT_PATH");
        if (raw_env == nullptr) {
            std::cerr << "Set JCN_PJRT_PATH to discover PJRT plugins" << std::endl;
            return;
        }

        std::string raw_path = std::string(raw_env) + "/pjrt";
        const PJRT_Api* cuda_pjrt_api = nullptr;

        try {
            struct stat st;
            if (stat(raw_path.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
                std::cerr << "Invalid PJRT plugin directory: " << raw_path << std::endl;
                return;
            }

            DIR* dir = opendir(raw_path.c_str());
            if (!dir) {
                std::cerr << "Failed to open PJRT plugin directory: " << raw_path << std::endl;
                return;
            }

            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                if (entry->d_name[0] == '.') continue;

                std::string backend(entry->d_name);
                std::string backend_dir = raw_path + "/" + backend;

                struct stat backend_st;
                if (stat(backend_dir.c_str(), &backend_st) != 0 || !S_ISDIR(backend_st.st_mode)) {
                    continue;
                }

                std::string plugin_path = backend_dir + "/pjrt_plugin.so";
                if (access(plugin_path.c_str(), R_OK) != 0) {
                    continue;
                }

                absl::StatusOr<const PJRT_Api*> status_or_api =
                    pjrt::LoadPjrtPlugin(backend, plugin_path);

                if (status_or_api.ok()) {
                    logger.log(LogLevel::INFO, "Loaded PJRT plugin " + backend);
                    if (backend == "cuda") {
                        cuda_pjrt_api = status_or_api.value();
                    }
                } else {
                    std::cerr << "Failed to load PJRT plugin " << backend
                            << ": " << status_or_api.status().ToString() << std::endl;
                }
            }

            closedir(dir);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load PJRT plugins: " << e.what() << std::endl;
        }

        if (cuda_pjrt_api != nullptr) {
            int oeq_rc = chemtrain_register_openequivariance_xla_ffi(
                cuda_pjrt_api, "CUDA");
            if (oeq_rc != 0) {
                throw std::runtime_error(
                    "Failed to register OpenEquivariance XLA FFI handlers for CUDA");
            }
            if (RegisterCommunicationFfi(cuda_pjrt_api, "CUDA") != 0) {
                throw std::runtime_error(
                    "Failed to register chemtrain communication XLA FFI handlers for CUDA");
            }
        }

    }

    Runner::Runner(ConnectorConfig connector_config, bool initialize)
        : config(std::move(connector_config)) {

        if (initialize) {
            Runner::initialize();
        }

        // Singleton
        Logger logger = Logger::getlogger();

        absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_or_status;

        if (this->config.backend == "cpu") {

          xla::CpuClientOptions create_options;
          create_options.asynchronous = true;

          client_or_status = xla::GetPjRtCpuClient(create_options);

        } else {

            logger.log(LogLevel::INFO, "Initializing PjRtClient for backend '" + this->config.backend + "' with options:");
            logger.log(LogLevel::INFO, "  - Device: " + std::to_string(this->config.device));
            logger.log(LogLevel::INFO, "  - Memory fraction: " + std::to_string(this->config.memory_fraction));


            absl::flat_hash_map<std::string, xla::PjRtValueType> create_options = {
                {"memory_fraction", static_cast<float>(this->config.memory_fraction)},
                {"visible_devices", std::vector<int64_t>({this->config.device})},
            };

            // Initialize the possible backends in the libconnector file
            absl::StatusOr<bool> status_or_success = pjrt::IsPjrtPluginInitialized(this->config.backend);
            if (!status_or_success.ok()) {
                throw std::runtime_error("Failed to initialize PjRtClient: " + status_or_success.status().ToString());
            }

            if (!status_or_success.value()) {
                absl::Status status = pjrt::InitializePjrtPlugin(this->config.backend);
                if (!status.ok()) {
                    throw std::runtime_error("Failed to initialize PjRtClient: " + status.ToString());
                }
            }

            // Get the client
            client_or_status = xla::GetCApiClient(this->config.backend, create_options);

        }

        if (!client_or_status.ok()) {
            throw std::runtime_error("Failed to initialize PjRtClient: " + client_or_status.status().ToString());
        }

        client = std::move(client_or_status).value();

        // Determine the index into addressable_devices() to use for buffer allocation.
        // When visible_devices filtering is applied, addressable_devices() may be remapped.
        pjrt_device_index_ = 0;
        absl::Span<xla::PjRtDevice* const> addressable = client->addressable_devices();
        if (addressable.empty()) {
            throw std::runtime_error("PjRtClient has no addressable devices");
        }

        for (int i = 0; i < addressable.size(); ++i) {
            if (addressable[i]->id() == this->config.device) {
                pjrt_device_index_ = i;
                break;
            }
        }

        logger.log(
            LogLevel::INFO,
            "Using addressable device index " + std::to_string(pjrt_device_index_) +
                " (requested id=" + std::to_string(this->config.device) +
                ", actual id=" + std::to_string(addressable[pjrt_device_index_]->id()) + ")"
        );

        // Print devices
        absl::Span<xla::PjRtDevice* const> devices = client->devices();
        std::string device_list = "";
        for (int i = 0; i < devices.size(); i++) {
            device_list += std::string(devices[i]->ToString()) + ",";
	    }
	    logger.log(LogLevel::INFO, "Found devices [" + device_list + "]");

    }


    ModelProperties Runner::load_model(ModelConfig config) {
        // Singleton
        Logger logger = Logger::getlogger();

        newton = config.newton;
        communication_callbacks = config.communication;

        model = std::make_unique<jcn::Model>();

        // Deserialize the protobuffer
        if (config.model.empty()) {
            throw std::runtime_error("Cannot load model: Model file is empty.");
        }

        if (!model->ParseFromString(config.model)) {
            throw std::runtime_error("Cannot load model: Model file is invalid or corrupted.");
        }

        logger.log(
            LogLevel::DEBUG,
            "Model communication: enabled=" +
                std::to_string(model->uses_communication()) +
                ", width=" +
                std::to_string(model->communication_buffer_width()) +
                ", newton=" + std::to_string(newton) +
                ", neighbor_orders=[" +
                (model->neighbor_list().nbr_order_size() > 0
                     ? std::to_string(model->neighbor_list().nbr_order(0))
                     : "missing") +
                ", " +
                (model->neighbor_list().nbr_order_size() > 1
                     ? std::to_string(model->neighbor_list().nbr_order(1))
                     : "missing") +
                "]");

        // Pass the mlir module to the compiler
        compiler = std::make_unique<Compiler>(model->mlir_module(),
                                              false);

        // Extract exported quantity keys from the model proto and pass to atom_builder
        std::vector<std::string> quantities;
        for (int i = 0; i < model->quantities_size(); ++i) {
            quantities.push_back(model->quantities(i));
        }
        atom_builder = std::make_unique<AtomBuilder>(config.atom_multiplier, config.newton, quantities);

        // Read out statistics required for the neighbor lists
        std::vector<std::string> statistics_keys;
        for (int i = 0; i < model->neighbor_list().statistics_keys_size(); i++) {
            statistics_keys.push_back(model->neighbor_list().statistics_keys(i));
        }

        // Select from the available neighbor list types
        switch (model->neighbor_list().type()) {
            case jcn::Model::SIMPLE_SPARSE:
                neighbor_list = std::make_unique<SimpleSparseNeighborList>(
                    statistics_keys
                );
                neighbor_list->initialize(config.neighbor_list_multipliers);

                logger.log(LogLevel::INFO, "Initialize SimpleSparseNeighborList");
                break;
            case jcn::Model::SIMPLE_DENSE:
                neighbor_list = std::make_unique<SimpleDenseNeighborList>(
                    statistics_keys
                );
                neighbor_list->initialize(config.neighbor_list_multipliers);

                logger.log(LogLevel::INFO, "Initialize SimpleDenseNeighborList");
                break;
            case jcn::Model::DEVICE_SPARSE:
                neighbor_list = std::make_unique<DeviceSparseNeighborList>(
                    statistics_keys
                );
                neighbor_list->initialize(config.neighbor_list_multipliers);

                logger.log(LogLevel::INFO, "Initialize DeviceSparseNeighborList");
                break;
            default:
                throw std::runtime_error(
                    "Unknown neighbor list type: "
                    + std::to_string(model->neighbor_list().type())
                );
        }

        return get_model_properties();

    }

    Results Runner::compute_forces(
        int lnum, int gnum, double **x, double **f, int *type, int inum,
        int *ilist, int *numneigh, int **firstneigh, bool list_changed,
        bool allow_recompile
    ) {

        // Singleton
        Logger logger = Logger::getlogger();

        int max_trials = 10;
        bool recompiled = false;

        for (int i = 0; i < max_trials; i++) {

            auto trial_start = std::chrono::high_resolution_clock::now();

            // First we build the domain and the neighbor list, then we can
            // determine the input shapes to the program

            AtomShapes atoms = atom_builder->get_shapes(lnum, gnum, allow_recompile);

            NeighborListShapes neighbors = neighbor_list->get_neighbor_list_shapes(
                atoms.n_atoms, inum, ilist, numneigh, allow_recompile);

            // Now we have all shapes setup to build the module if required.
            // If the module tried to recompile but failed due to disabled
            // recompilation, it will try again in the next call due to the
            // flag recompilation_required.
            // If recompilation is not necessary but allowed, it will depend
            // on how much the buffers are filled.
            recompilation_required |= !executable || atoms.reallocate || neighbors.reallocate;
            if (recompilation_required && allow_recompile) {
                recompiled |= true; // Track for statistics whether recompilation was necessary

                logger.log(LogLevel::INFO, "Recompilation necessary");

                compiler->compile(
                    atoms.n_atoms, neighbors.graph_shapes, neighbors.graph_types);

                absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> executable_or_status = client->CompileAndLoad(
                    compiler->module(), compile_options);

                if (!executable_or_status.ok()) {
                    throw std::runtime_error("Failed to compile: " + executable_or_status.status().ToString());
                }

                executable = std::move(executable_or_status).value();

                // Print a cost analysis of the exectuable
                absl::StatusOr<absl::flat_hash_map<std::string, xla::PjRtValueType>> cost_analysis;
                cost_analysis = executable->GetCostAnalysis();
                if (cost_analysis.ok()) {
                    const absl::flat_hash_map<std::string, xla::PjRtValueType>& cost_map = cost_analysis.value();

                    auto it = cost_map.find("flops");
                    if (it != cost_map.end()) {
                        if (const float* flops = std::get_if<float>(&it->second)) {
                            flops_ = *flops;
                            logger.log(LogLevel::INFO,
                                "Cost analysis: " + std::to_string(*flops) + " flops."
                            );
                        } else {
                            std::cerr << "Error: 'flops' is not a float type" << std::endl;
                        }
                    } else {
                        std::cerr << "Error: 'flops' key not found in cost_map" << std::endl;
                    }
                } else {
                    std::cerr << "Failed to get cost analysis: " << cost_analysis.status().ToString() << std::endl;
                }

                recompilation_required = false; // Reset the recompilation flag

            } else if (recompilation_required) {
                throw jcn::RecompilationRequired(
                    "Recompilation required but not allowed. Please set allow_recompile to true.");
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Only transfer new data to the GPU if necessary
            bool update = (recompiled || list_changed);

            // Now we have to create the buffers, i.e., copy the data onto
            // the device
            std::vector<xla::PjRtBuffer*> buffer_ptrs = atom_builder->build_domain(client.get(), pjrt_device_index_, lnum, gnum, x, type);

            // TODO: We have to add the gnum option to the neighbor list.
            //       This is only a workaround for the sparse neighbor list
            //       which includes the ghost atoms as senders.
            std::vector<xla::PjRtBuffer*> graph_buffers = neighbor_list->build_graph(
                client.get(), pjrt_device_index_, inum, ilist, numneigh, firstneigh, update);
            buffer_ptrs.insert(buffer_ptrs.end(), graph_buffers.begin(), graph_buffers.end());

            std::vector<std::vector<xla::PjRtBuffer*>> arg_handles = {buffer_ptrs};

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            logger.log(LogLevel::DEBUG, "Time taken for buffer creation: " + std::to_string(duration.count()) + " seconds");

            // Check if arg_handles is correctly populated
            if (arg_handles.empty() || arg_handles[0].empty()) {
                throw std::runtime_error("arg_handles is empty or not properly populated");
            }

            xla::ExecuteContext execute_context;
            CommunicationContext communication_context(
                communication_callbacks, model->uses_communication(),
                &communication_workspace_);

            absl::Status context_status =
                AddCommunicationContextToExecuteContext(
                    &execute_context, &communication_context);

            if (!context_status.ok()) {
                throw std::runtime_error(
                    "Failed to initialize communication execution context: " +
                    context_status.ToString());
            }

            xla::ExecuteOptions execute_options;
            execute_options.context = &execute_context;

            start = std::chrono::high_resolution_clock::now();

            // Use std::async to execute the function asynchronously
            std::future<absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>>> future_results =
                std::async(std::launch::async, [&]() {
                    auto results = executable->Execute(
                        absl::Span<const std::vector<xla::PjRtBuffer*>>(arg_handles),
                        execute_options
                    );
                    if (!results.ok()) return results;
                    // Execute only enqueues GPU work. Keep the rendezvous loop
                    // alive until the FFI calls and all dependent work finish.
                    for (const auto& replica : results.value()) {
                        for (const auto& buffer : replica) {
                            absl::Status ready = buffer->GetReadyFuture().Await();
                            if (!ready.ok()) return decltype(results)(ready);
                        }
                    }
                    return results;
                });

            // PJRT invokes FFI handlers on its worker. Service their staged
            // requests here so LAMMPS and MPI are entered only by this thread.
            while (future_results.wait_for(std::chrono::milliseconds(1)) !=
                   std::future_status::ready) {
                communication_context.ServiceOne();
            }
            while (communication_context.ServiceOne()) {}

            // Wait for the results to be ready
            absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>> results = future_results.get();

            if (!results.ok()) {
                throw std::runtime_error("Failed to execute: " + results.status().ToString());
            }

            // Now we have to copy the results back to the host
            std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results_buffers = std::move(results).value();

            // Sort out the results buffers. Map statistics after the exported quantities.
            // The atom_builder carries the exported `quantities` keys so we can
            // determine the offset into the returned results.
            std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics;
            int offset = static_cast<int>(atom_builder->get_quantities().size());
            for (int i = 0; i < neighbor_list->statistics_keys.size(); i++) {
                statistics.emplace(
                    neighbor_list->statistics_keys[i],
                    std::move(results_buffers[0][i + offset])
                );
            }

            bool success = neighbor_list->evaluate_statistics(
                std::move(statistics), allow_recompile);

            end = std::chrono::high_resolution_clock::now();
            duration = end - start;

            logger.log(LogLevel::DEBUG, "Time taken for computation: " + std::to_string(duration.count()) + " seconds");

            // Write back the results
            double potential = atom_builder->evaluate_domain(
                success, lnum, gnum, f, results_buffers);

            auto trial_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> trial_duration = trial_end - trial_start;

            logger.log(LogLevel::DEBUG, "Time taken for trial: " + std::to_string(trial_duration.count()) + " seconds");

            results_buffers.clear();

            Results compute_results;
            compute_results.potential = potential;
            compute_results.stats.flops = flops_;
            compute_results.stats.recompiled = recompiled;

            // Finished
            if (success) return compute_results;

        }

        throw std::runtime_error("Failed to compute forces after " + std::to_string(max_trials) + " trials");

    }

    ModelProperties Runner::get_model_properties() {
        // Singleton
        Logger logger = Logger::getlogger();

        ModelProperties properties;

        if (!model) {
            throw std::runtime_error("Model is not initialized");
        }

        // Sufficient number of ghost atomus must be communicated.
        // The communication cutoff depends on the number of message passing
        // steps which effectively increase the cutoff distance.
        int multiplier;
        if (newton) {
            multiplier = model->neighbor_list().nbr_order()[0];
        } else {
            multiplier = model->neighbor_list().nbr_order()[1];
        }
        properties.comm_dist = multiplier * model->neighbor_list().cutoff();
        properties.communication_buffer_width =
            model->communication_buffer_width();

        if (model->has_unit_style()) {
            properties.unit_style = model->unit_style().c_str();
        } else {
            properties.unit_style = "real"; // Define this as default
        }

        switch (model->neighbor_list().type()) {
            case jcn::Model::SIMPLE_SPARSE:
            case jcn::Model::SIMPLE_DENSE:
                // Neighbor list cutoff must be larger than the model cutoff
                properties.cutoff = model->neighbor_list().cutoff();

                if (multiplier > 1) {
                    // Ghost atoms only required if more than the next neighbor
                    // can affect the local energy of a particle
                    properties.neighbor_list.include_ghosts = true;
                    logger.log(LogLevel::INFO,
                        "Include ghosts: " + std::to_string(properties.neighbor_list.include_ghosts)
                    );
                };
                if (model->neighbor_list().has_half_list()) {
                    properties.neighbor_list.half_list = model->neighbor_list().half_list();
                    logger.log(LogLevel::INFO,
                        "Use half list only " + std::to_string(properties.neighbor_list.half_list)
                    );
                };

                break;
            case jcn::Model::DEVICE_SPARSE:
                // Does not specify a cutoff for the particles as neighbor
                // list is computed on the device
                properties.cutoff = 0.0;

                break;
        }

        logger.log(LogLevel::INFO,
            std::string("Model properties:") +
            "\n\t-Cutoff: " + std::to_string(properties.cutoff) +
            "\n\t-Com. distance: " + std::to_string(properties.comm_dist) +
            "\n\t-Unit style: " + properties.unit_style
        );

        return properties;
    }

} // namespace jcn
