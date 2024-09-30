//
// Created by Paul Fuchs on 24.09.24.
//

#include "xla/literal.h"

#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

namespace jcn {

    struct NeighborList {
        // Return the shapes in the neighbor list builder
        std::vector<std::vector<int64_t>> graph_shapes;
        std::vector<xla::PrimitiveType> graph_types;

        // Let the neighbor list keep track whether reallocation is needed
        // in case of an increased number of atoms or neighbors
        bool reallocate;

        // The actual values of the
        std::vector<xla::Literal*> graph_values;
    };

    class SimpleSparseNeighborList {

        public:
            SimpleSparseNeighborList(float edge_multiplier);
            ~SimpleSparseNeighborList() = default;

            // TODO: Documentation
            //
            // @param ilist: Index of the sender atom
            // @param numneigh: Number of neighbors for the sender atom
            // @param firstneigh: Start of the neighbor list for the first atom
            NeighborList build_neighbor_list(
                int max_atoms, int inum, int *ilist, int *numneigh, int **firstneigh,
                bool list_changed);

        private:
            float edge_multiplier;

            std::unique_ptr<xla::Literal> senders_literal;
            std::unique_ptr<xla::Literal> receivers_literal;

            int n_edges = 0;

    };

} // namespace jcn

#endif //GRAPH_BUILDER_H
