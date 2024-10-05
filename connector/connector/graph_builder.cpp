//
// Created by Paul Fuchs on 24.09.24.
//

#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/literal_util.h"

#include <chrono>
#include <bitset>
#include <iostream>

#include "graph_builder.h"
#include "pjrt.h"

#include "pjrt.h"

namespace jcn {

    void SimpleSparseNeighborList::initialize(std::vector<float> multipliers) {
        edge_multiplier = multipliers[0];
    }


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


    void DeviceSparseNeighborList::initialize(std::vector<float> multipliers) {
        edge_multiplier = multipliers[0];
        capacity_multiplier = multipliers[1];
    }


    bool DeviceSparseNeighborList::adjust_dimension(std::unique_ptr<xla::Literal>& cells, int size, xla::PrimitiveType type) {

        // Check if evaluation of the statistics increased the number of cells
        if (cells && cells->shape().dimensions(0) >= size) return false;

        std::cout << "Reallocate literal" << std::endl;

        // Reallocate the cells
        xla::Shape shape = xla::ShapeUtil::MakeShape(
            type, absl::Span<const int64_t>{size});

        cells = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(shape));
        // cells->PopulateWithValue(0); // Value is not important

        return true;
    }


    NeighborListShapes DeviceSparseNeighborList::get_neighbor_list_shapes(
         int max_atoms, int inum, int* numneigh) {
         // We pass the neighbor list of LAMMPS

         // No reallocation necessary if buffers sufficiently large
         bool reallocate = false;

         // Check the cell dimensions (works also for the capacity and senders)
         reallocate |= adjust_dimension(xcells_lit, n_cells_x, xla::PRED);
         reallocate |= adjust_dimension(ycells_lit, n_cells_y, xla::PRED);
         reallocate |= adjust_dimension(zcells_lit, n_cells_z, xla::PRED);
         reallocate |= adjust_dimension(capacity_lit, capacity, xla::PRED);
         reallocate |= adjust_dimension(senders_lit, n_edges, xla::S32);


         std::vector<std::vector<int64_t>> graph_shapes = {
             {n_cells_x}, {n_cells_y}, {n_cells_z}, {capacity}, {n_edges}
         };
         std::vector<xla::PrimitiveType> graph_types = {
             xla::PRED, xla::PRED, xla::PRED, xla::PRED, xla::S32
         };

         return NeighborListShapes{graph_shapes, graph_types, reallocate};

    }


    std::vector<xla::PjRtBuffer*> DeviceSparseNeighborList::build_graph(
        xla::PjRtClient* client, int device_id, int inum, int *ilist,
        int *numneigh, int **firstneigh, bool update) {

        if (update) {
            // Clear old buffers
            if (senders_buffer) {
                xcells_buffer->Delete();
                ycells_buffer->Delete();
                zcells_buffer->Delete();
                capacity_buffer->Delete();
                senders_buffer->Delete();
            }

            // Create buffers (value is not important)
            xcells_buffer = create_buffer(client, device_id, xcells_lit.get());
            ycells_buffer = create_buffer(client, device_id, ycells_lit.get());
            zcells_buffer = create_buffer(client, device_id, zcells_lit.get());
            capacity_buffer = create_buffer(client, device_id, capacity_lit.get());
            senders_buffer = create_buffer(client, device_id, senders_lit.get());

        }

        std::vector<xla::PjRtBuffer*> buffer_ptrs;

        buffer_ptrs.push_back(xcells_buffer.get());
        buffer_ptrs.push_back(ycells_buffer.get());
        buffer_ptrs.push_back(zcells_buffer.get());
        buffer_ptrs.push_back(capacity_buffer.get());
        buffer_ptrs.push_back(senders_buffer.get());

        return buffer_ptrs;
    }


    bool DeviceSparseNeighborList::evaluate_statistics(
        std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>& results) {

        bool success = true;

        absl::StatusOr<std::shared_ptr<xla::Literal>> min_cell_capacity = results[0][2]->ToLiteralSync();
        absl::StatusOr<std::shared_ptr<xla::Literal>> cell_too_small = results[0][3]->ToLiteralSync();
        absl::StatusOr<std::shared_ptr<xla::Literal>> min_neighbors = results[0][4]->ToLiteralSync();

        if (!min_cell_capacity.ok() || !cell_too_small.ok() || !min_neighbors.ok()) {
            throw std::runtime_error("Failed to convert buffer to literal");
        }

        std::cout << "Returned statistics:" << std::endl \
                  << "- Min cell capacity: " << min_cell_capacity.value()->data<int>().data()[0] << std::endl \
                  << "- Cell too small: " << cell_too_small.value()->data<int>().data()[0] << std::endl \
                  << "- Min neighbors: " << min_neighbors.value()->data<int>().data()[0] << std::endl;

        // TODO
        // Get back the directions with too small cell sizes
        // std::bitset<3> binary(number);

        // Adjust the capcities of cells and neighborlist
        int req_cell_capacity = min_cell_capacity.value()->data<int>().data()[0];
        if (req_cell_capacity > capacity) {
            capacity = static_cast<int>(std::ceil(req_cell_capacity * capacity_multiplier));
            success = false;
        }

        int req_nbrs_capacity = min_neighbors.value()->data<int>().data()[0];
        if (req_nbrs_capacity > n_edges) {
            n_edges = static_cast<int>(std::ceil(req_nbrs_capacity * edge_multiplier));
            success = false;
        }

        // Returns whether rerun with bigger capacities is necessary
        return success;

    }

} // namespace jcn
