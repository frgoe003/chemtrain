//
// Created by Paul Fuchs on 24.09.24.
//

#include "xla/literal_util.h"

#include <chrono>

#include "graph_builder.h"

namespace jcn {

    SimpleSparseNeighborList::SimpleSparseNeighborList(float edge_multiplier)
        : edge_multiplier(edge_multiplier) {}

    NeighborList SimpleSparseNeighborList::build_neighbor_list(
        int max_atoms, int inum, int *ilist, int *numneigh, int **firstneigh) {
        // We pass the neighbor list of LAMMPS

        // No reallocation necessary if buffers sufficiently large
        bool reallocate = false;
        if (!senders_literal || !receivers_literal) {
            reallocate = true;
        }

        auto start = std::chrono::high_resolution_clock::now();

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
        std::fill(senders_data + edge_counter, senders_data + n_edges, max_atoms);
        std::fill(receivers_data + edge_counter, receivers_data + n_edges, max_atoms);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Time taken for neighborlist array creation: " << duration.count() << " seconds" << std::endl;

//        for (int i = 0; i < edge_counter; i++) {
//            std::cout << "Edge " << i << ": " << senders(i) << " -> " << receivers(i) << std::endl;
//        }

        // Create the literals. Since these are not copyable, we only pass
        // the unique pointers
//        std::unique_ptr<xla::Literal> senders_literal = std::make_unique<xla::Literal>(xla::LiteralUtil::CreateFromArray(senders));
//        std::unique_ptr<xla::Literal> receivers_literal = std::make_unique<xla::Literal>(xla::LiteralUtil::CreateFromArray(receivers));


        std::vector<xla::Literal*> graph_values;
        graph_values.push_back(senders_literal.get());
        graph_values.push_back(receivers_literal.get());

        std::vector<std::vector<int64_t>> graph_shapes = {{n_edges}, {n_edges}};
        std::vector<xla::PrimitiveType> graph_types = {xla::S32, xla::S32};

        return NeighborList{graph_shapes, graph_types, reallocate, std::move(graph_values)};

    };

} // namespace jcn
