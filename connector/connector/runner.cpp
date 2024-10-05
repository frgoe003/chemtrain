//
// Created by Paul Fuchs on 24.09.24.
//

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>  // For std::unique_ptr
#include <dlfcn.h>

#include "runner.h"
#include "compiler.h"
#include "libconnector.h"
#include "domain.h"
#include "pjrt.h"

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


namespace jcn {

    void initialize() {

        // TODO: Make this more general
        std::cout << "Try to load GPU Plugin" << std::endl;

        // Load the CUDA plugin from JAX
        // TODO: Currently only finds devices if the pre-built plugin is used.
        absl::StatusOr<const PJRT_Api*> status_or_api = pjrt::LoadPjrtPlugin(
            "cuda",
            "/home/paul/miniconda3/envs/chemtrain/lib/python3.11/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so");

        if (!status_or_api.ok()) {
            LOG(INFO) << "Failed to load CUDA plugin: " << status_or_api.status();
            return;
        }

        std::cout << "Try to initiliaze plugin" << std::endl;
        if (!pjrt::IsPjrtPluginInitialized("cuda").value()) {
            std::cerr << "Initialize CUDA plugin" << std::endl;

            pjrt::InitializePjrtPlugin("cuda");
        }
        std::cout << "Finished initialization" << std::endl;


    }


    Runner::Runner(ConnectorConfig config) :
        atom_builder(1.25),
        compiler(config.mlir_module)
    {

        // Select from the available neighbor list types
        if (config.neighbor_list_type == "SimpleSparseNeighborList") {
            neighbor_list = std::make_unique<SimpleSparseNeighborList>();
            neighbor_list->initialize(config.neighbor_list_multipliers);
        } else if (config.neighbor_list_type == "DeviceSparseNeighborList") {
            neighbor_list = std::make_unique<DeviceSparseNeighborList>();
            neighbor_list->initialize(config.neighbor_list_multipliers);
        } else {
            throw std::runtime_error("Unknown neighbor list type: " + config.neighbor_list_type);
        }

        // TODO: Maybe move this stuff to some better place
        initialize();

        // Initialize the possible backends in the libconnector file
        std::cout << "Try to create client" << std::endl;
        absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_or_status = xla::GetCApiClient(config.backend);
        if (!client_or_status.ok()) {
            throw std::runtime_error("Cannot create client: " + client_or_status.status().ToString());
            client = xla::GetTfrtCpuClient(/*asynchronous=*/true).value();
        } else {
            client = std::move(client_or_status).value();
        }

    }

    double Runner::compute_forces(
        int inum, int gnum, double **x, double **f, int *type, int *ilist, int *numneigh, int **firstneigh, bool list_changed) {

            int max_trials = 10;

            for (int i = 0; i < max_trials; i++) {

                auto trial_start = std::chrono::high_resolution_clock::now();

                // First we build the domain and the neighbor list, then we can
                // determine the input shapes to the program

                AtomShapes atoms = atom_builder.get_shapes(inum, gnum);
                NeighborListShapes neighbors = neighbor_list->get_neighbor_list_shapes(
                    atoms.n_atoms, inum, numneigh);

                // Now we have all shapes setup to build the module if required
                if (!executable || atoms.reallocate || neighbors.reallocate ) {
                     xla::XlaComputation callable = compiler.compile(
                        atoms.n_atoms, neighbors.graph_shapes, neighbors.graph_types);

                    // No idea what to specify here...
                    xla::CompileOptions compile_options;

                    absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> executable_or_status = client->Compile(callable, compile_options);

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
                std::vector<xla::PjRtBuffer*> buffer_ptrs = atom_builder.build_domain(client.get(), 0, inum, gnum, x, type);

                std::vector<xla::PjRtBuffer*> graph_buffers = neighbor_list->build_graph(
                    client.get(), 0, inum, ilist, numneigh, firstneigh, update);
                buffer_ptrs.insert(buffer_ptrs.end(), graph_buffers.begin(), graph_buffers.end());

                std::vector<std::vector<xla::PjRtBuffer*>> arg_handles = {buffer_ptrs};

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - start;
                std::cout << "Time taken for buffer creation: " << duration.count() << " seconds" << std::endl;

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

                // Iterate through the results_buffers and print the shapes
for (const auto& buffer_vector : results_buffers) {
    std::cout << "Print contents" << std::endl;
    for (const auto& buffer : buffer_vector) {
        if (buffer) {
            // Use ToLiteralSync to get the shape information
            absl::StatusOr<std::shared_ptr<xla::Literal>> literal_or_status = buffer->ToLiteralSync();
            if (literal_or_status.ok()) {
                auto literal = literal_or_status.value();
                std::cout << "Buffer shape: " << literal->shape().ToString() << std::endl;
            } else {
                std::cout << "Failed to get literal: " << literal_or_status.status().ToString() << std::endl;
            }
        } else {
            std::cout << "Buffer is null" << std::endl;
        }
    }
}

                bool success = neighbor_list->evaluate_statistics(results_buffers);

                end = std::chrono::high_resolution_clock::now();
                duration = end - start;
                std::cout << "Time taken for computation: " << duration.count() << " seconds" << std::endl;


                // Write back the results
                double potential = atom_builder.evaluate_domain(
                    success, inum, f, results_buffers);

                auto trial_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> trial_duration = trial_end - trial_start;
                std::cout << "Time taken for trial: " << trial_duration.count() << " seconds" << std::endl;

                results_buffers.clear();

                // Finished
                if (success) return potential;

            }

        throw std::runtime_error("Failed to compute forces after " + std::to_string(max_trials) + " trials");

      }

} // namespace jcn
