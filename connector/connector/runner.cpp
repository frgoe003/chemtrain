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
        neighbor_list(1.5), // Hard-coded the multipliers for now
        atom_builder(1.5),
        compiler(config.mlir_module)
    {

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

            // First we build the domain and the neighbor list, then we can
            // determine the input shapes to the program

            Atoms atoms = atom_builder.build_domain(inum, gnum, x, type);
            NeighborList neighbors = neighbor_list.build_neighbor_list(
                atoms.n_atoms, inum, ilist, numneigh, firstneigh, list_changed);

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

            // TODO: Seems to be necessary somehow. Find out what it does
            std::unique_ptr<xla::Literal> platform_index = std::make_unique<xla::Literal>(
                xla::LiteralUtil::CreateR0<int32_t>(0)
            );

            // Now we have the executable, we have to move the data
            std::vector<xla::Literal*> literals;
            // literals.push_back(std::move(platform_index));
            literals.push_back(atoms.positions);
            literals.push_back(atoms.species);
            literals.push_back(atoms.ghost_mask);

            for (int i = 0; i < neighbors.graph_values.size(); i++) {
                literals.push_back(std::move(neighbors.graph_values[i]));
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Now we have to create the buffers, i.e., copy the data onto
            // the device
            std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers;
            std::vector<xla::PjRtBuffer*> buffer_ptrs;
            for (int i = 0; i < literals.size(); i++) {
                // TODO: Make the addressable device a parameter
                absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> input_buffer = client->BufferFromHostBuffer(
                    literals[i]->untyped_data(),
                    literals[i]->shape().element_type(),
                    literals[i]->shape().dimensions(),
                    std::optional<absl::Span<int64_t const>>{},
                    xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
                    []() { /* frees literal */ },
                    client->addressable_devices()[0]
                );

                if (!input_buffer.ok()) {
                    throw std::runtime_error("Failed to create buffer: " + input_buffer.status().ToString());
                }

                // There must be a better way than storing twice a reference
                // to the buffers
                buffer_ptrs.push_back(input_buffer.value().get());
                buffers.push_back(std::move(input_buffer).value());
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            std::cout << "Time taken for buffer creation: " << duration.count() << " seconds" << std::endl;


            std::vector<std::vector<xla::PjRtBuffer*>> arg_handles = {buffer_ptrs};

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

            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            std::cout << "Time taken for computation: " << duration.count() << " seconds" << std::endl;

            if (!results.ok()) {
                throw std::runtime_error("Failed to execute: " + results.status().ToString());
            }


            // Now we have to copy the results back to the host
            std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results_buffers = std::move(results).value();
            absl::StatusOr<std::shared_ptr<xla::Literal>> force_literal = results_buffers[0][0]->ToLiteralSync();
            absl::StatusOr<std::shared_ptr<xla::Literal>> energy_literal = results_buffers[0][1]->ToLiteralSync();

            if (!force_literal.ok() || !energy_literal.ok()) {
                throw std::runtime_error("Failed to copy results: " + force_literal.status().ToString() + " " + energy_literal.status().ToString());
            }

            // Write back the results
            double potential = atom_builder.evaluate_domain(
                inum, f, force_literal.value(), energy_literal.value());

            // Destroy the buffers explicitly
            for (auto& buffer : buffers) {
                buffer->Delete();
            }
            results_buffer->Delete();

            return potential;

      }

} // namespace jcn
