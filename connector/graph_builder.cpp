//
// Created by Paul Fuchs on 24.09.24.
//

#include "xla/literal_util.h"

#include "graph_builder.h"

namespace jcn {

    SimpleSparseNeighborList::SimpleSparseNeighborList(float atom_multiplier, float edge_multiplier)
        : atom_multiplier(atom_multiplier), edge_multiplier(edge_multiplier) {}

    NeighborList SimpleSparseNeighborList::build_neighbor_list(
        int max_atoms, int inum, int *ilist, int *numneigh, int **firstneigh) {
        // We pass the neighbor list of LAMMPS

        // No reallocation necessary if buffers sufficiently large
        bool reallocate = false;

        // Count the number of edges
        int current_edges = 0;
        for (int i = 0; i < inum; i++) {
            current_edges += numneigh[i];
        }

        // Check if reallocation is necessary
        if (current_edges > n_edges) {
            n_edges = current_edges * edge_multiplier;
            reallocate = true;
        }

        // Allocate sender and receiver literals
        xla::Array<int> senders(std::vector<int64_t>{n_edges}, max_atoms);
        xla::Array<int> receivers(std::vector<int64_t>{n_edges}, max_atoms);

        // Fill in the sender and receiver values
        int edge_counter = 0;
        for (int i = 0; i < inum; i++) {
            for (int j = 0; j < numneigh[i]; j++) {
                // Make the edges undirected but add them only once.
                // The ghost atom indices are larger than the local atom indices
                if (i < j) {
                    senders(edge_counter) = ilist[i];
                    receivers(edge_counter) = firstneigh[i][j];
                    edge_counter++;
                }
            }
        }

        // Create the literals. Since these are not copyable, we only pass
        // the unique pointers
        std::unique_ptr<xla::Literal> senders_literal = std::make_unique<xla::Literal>(xla::LiteralUtil::CreateFromArray(senders));
        std::unique_ptr<xla::Literal> receivers_literal = std::make_unique<xla::Literal>(xla::LiteralUtil::CreateFromArray(receivers));

        std::vector<std::unique_ptr<xla::Literal>> graph_values;
        graph_values.push_back(std::move(senders_literal));
        graph_values.push_back(std::move(receivers_literal));

        std::vector<std::vector<int64_t>> graph_shapes = {{n_edges, n_edges}};
        std::vector<xla::PrimitiveType> graph_types = {xla::S32, xla::S32};

        return NeighborList{graph_shapes, graph_types, reallocate, std::move(graph_values)};

    };

} // namespace jcn
