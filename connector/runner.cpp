//
// Created by Paul Fuchs on 24.09.24.
//

#include <iostream>
#include <string>
#include <vector>
#include <memory>  // For std::unique_ptr
#include <dlfcn.h>

#include "runner.h"
#include "compiler.h"
#include "libconnector.h"

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


    }

    Atoms AtomBuilder::build_domain(int inum, int gnum, double **x, int *type) {

        // If number of atoms in domain (including ghost) exceeds the allocated
        // buffers
        bool reallocate = false;

        if ((inum + gnum) > max_atoms) {
            max_atoms = atom_multiplier * (inum + gnum);
            reallocate = true;
        }

        xla::Array<float> positions(std::vector<int64_t>{max_atoms, 3}, 0);
        xla::Array<int> species(std::vector<int64_t>{max_atoms}, 0);
        xla::Array<bool> ghost_mask(std::vector<int64_t>{max_atoms}, 0);

        // Collect data for all local atoms and ghost atoms
        for (int i = 0; i < inum + gnum; i++) {
            for (int j = 0; j < 3; j++) {
                positions(i, j) = x[i][j];
            }
            // If atom is local atom, mark it in the mask
            if (i < inum) {
                ghost_mask(i) = true;
            }
            // Read out the species values and correct for 0-based type definition
            species(i) = type[i] - 1;
        }

        // Create literals
        std::unique_ptr<xla::Literal> positions_literal = std::make_unique<xla::Literal>(xla::LiteralUtil::CreateFromArray(positions));
        std::unique_ptr<xla::Literal> species_literal = std::make_unique<xla::Literal>(xla::LiteralUtil::CreateFromArray(species));
        std::unique_ptr<xla::Literal> ghost_mask_literal = std::make_unique<xla::Literal>(xla::LiteralUtil::CreateFromArray(ghost_mask));

        return Atoms{max_atoms, reallocate, std::move(positions_literal), std::move(species_literal), std::move(ghost_mask_literal)};

    }

    double AtomBuilder::evaluate_domain(int inum, double **f, std::shared_ptr<xla::Literal> forces, std::shared_ptr<xla::Literal> potential) {
        absl::Span<float> force_data = forces->data<float>();
        absl::Span<float> potential_data = potential->data<float>();

        // We skip all ghost atoms and padded atoms and only write back forces
        // on the real atoms
        for (int i = 0; i < inum; i++) {
            for (int j = 0; j < 3; j++) {
                f[i][j] = static_cast<double>(force_data[i * 3 + j]);
            }
        }

        return (double) potential_data[0];

    }

    Runner::Runner(ConnectorConfig config) :
        neighbor_list(1.5, 1.5), // Hard-coded the multipliers for now
        atom_builder(1.5),
        compiler(config.mlir_module)
    {

        // Initialize the possible backends in the libconnector file
        absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_or_status = xla::GetCApiClient(config.backend);
        if (!client_or_status.ok()) {
            std::cerr << "Cannot create client: " << client_or_status.status().ToString() << std::endl;
            client = xla::GetTfrtCpuClient(/*asynchronous=*/true).value();
        } else {
            client = std::move(client_or_status).value();
        }

    }

    double Runner::compute_forces(
        int inum, int gnum, double **x, double **f, int *type, int *ilist, int *numneigh, int **firstneigh) {

            // First we build the domain and the neighbor list, then we can
            // determine the input shapes to the program

            Atoms atoms = atom_builder.build_domain(inum, gnum, x, type);
            NeighborList neighbors = neighbor_list.build_neighbor_list(
                atoms.n_atoms, inum, ilist, numneigh, firstneigh);

            // Now we have all shapes setup to build the module if required
            if (!executable || atoms.reallocate ||neighbors.reallocate ) {
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

            // Now we have the executable, we have to move the data
            std::vector<std::unique_ptr<xla::Literal>> literals;
            literals.push_back(std::move(atoms.positions));
            literals.push_back(std::move(atoms.species));
            literals.push_back(std::move(atoms.ghost_mask));

            for (int i = 0; i < neighbors.graph_values.size(); i++) {
                literals.push_back(std::move(neighbors.graph_values[i]));
            }

            // Now we have to create the buffers, i.e., copy the data onto
            // the device
            std::vector<xla::PjRtBuffer*> buffers;
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

                buffers.push_back(std::move(input_buffer).value().get());
            }
            std::vector<std::vector<xla::PjRtBuffer*>> arg_handles = {buffers};

            // No idea what to specify here...
            xla::ExecuteOptions execute_options;

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
            absl::StatusOr<std::shared_ptr<xla::Literal>> force_literal = results_buffers[0][0]->ToLiteralSync();
            absl::StatusOr<std::shared_ptr<xla::Literal>> energy_literal = results_buffers[0][1]->ToLiteralSync();

            if (!force_literal.ok() || !energy_literal.ok()) {
                throw std::runtime_error("Failed to copy results: " + force_literal.status().ToString() + " " + energy_literal.status().ToString());
            }

            // Write back the results
            double potential = atom_builder.evaluate_domain(
                inum, f, force_literal.value(), energy_literal.value());

            return potential;

      }

} // namespace jcn
