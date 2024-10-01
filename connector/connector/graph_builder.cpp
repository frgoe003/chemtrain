//
// Created by Paul Fuchs on 24.09.24.
//

#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/literal_util.h"

#include <chrono>

#include "graph_builder.h"
#include "pjrt.h"

#include "pjrt.h"

namespace jcn {

    SimpleSparseNeighborList::SimpleSparseNeighborList(float edge_multiplier)
        : edge_multiplier(edge_multiplier) {}

    NeighborListShapes SimpleSparseNeighborList::get_neighbor_list_shapes(
        int max_atoms, int inum, int* numneigh) {
        // We pass the neighbor list of LAMMPS

        fill_value = max_atoms;

        // No reallocation necessary if buffers sufficiently large
        bool reallocate = false;
        if (!senders_literal || !receivers_literal) {
            reallocate = true;
        }

        // Count the number of edges
        int current_edges = 0;
        for (int i = 0; i < inum; i++) {
            current_edges += numneigh[i];
        }

        // Check if reallocation is necessary
        if (current_edges > n_edges) {
            std::cout << "Reallocation necessary " << std::endl;
            n_edges = static_cast<int>(std::ceil(current_edges * edge_multiplier));
            reallocate = true;
        }

        if (reallocate) {
            std::cout << "Reallocating to " << n_edges << " edges" << std::endl;
            xla::Shape shape = xla::ShapeUtil::MakeShape(
                xla::S32, absl::Span<const int64_t>{n_edges});

            senders_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(shape));
            receivers_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(shape));
        }

        std::vector<std::vector<int64_t>> graph_shapes = {{n_edges}, {n_edges}};
        std::vector<xla::PrimitiveType> graph_types = {xla::S32, xla::S32};

        return NeighborListShapes{graph_shapes, graph_types, reallocate};

    }


    std::vector<xla::PjRtBuffer*> SimpleSparseNeighborList::build_graph(
            xla::PjRtClient* client, int device_id, int inum, int *ilist, int *numneigh, int **firstneigh, bool update) {

        if (update) {

            // Clear old buffers
            if (senders_buffer) {
                senders_buffer->Delete();
                receivers_buffer->Delete();
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Only update the values if the shape or content of the neighbor list changed
            int* senders_data = senders_literal->data<int>().data();
            int* receivers_data = receivers_literal->data<int>().data();

            // Fill in the sender and receiver values
            int edge_counter = 0;
            for (int i = 0; i < inum; i++) {
                int num_neighbors = numneigh[i];
                int* ilist_ptr = &ilist[i];
                int* firstneigh_ptr = firstneigh[i];

                // Copy ilist[i] to senders_data
                std::fill(senders_data + edge_counter, senders_data + edge_counter + num_neighbors, ilist[i]);

                // Copy firstneigh[i] to receivers_data
                std::memcpy(receivers_data + edge_counter, firstneigh_ptr, num_neighbors * sizeof(int));

                edge_counter += num_neighbors;
            }

            // Fill in the invalid values
            std::fill(senders_data + edge_counter, senders_data + n_edges, fill_value);
            std::fill(receivers_data + edge_counter, receivers_data + n_edges, fill_value);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            std::cout << "Time taken for neighborlist array creation: " << duration.count() << " seconds" << std::endl;

            // Create buffers
            senders_buffer = create_buffer(client, device_id, senders_literal.get());
            receivers_buffer = create_buffer(client, device_id, receivers_literal.get());
        }

        // Return pointers to the buffers.
        std::vector<xla::PjRtBuffer*> buffer_ptrs;
        buffer_ptrs.push_back(senders_buffer.get());
        buffer_ptrs.push_back(receivers_buffer.get());

        return buffer_ptrs;

    };

} // namespace jcn
