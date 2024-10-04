//
// Created by Paul Fuchs on 24.09.24.
//

#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/literal_util.h"

#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

namespace jcn {

    struct NeighborListShapes {
        // Return the shapes in the neighbor list builder
        std::vector<std::vector<int64_t>> graph_shapes;
        std::vector<xla::PrimitiveType> graph_types;

        // Let the neighbor list keep track whether reallocation is needed
        // in case of an increased number of atoms or neighbors
        bool reallocate;
    };

    class GraphBuilder {
    public:
        GraphBuilder() = default;
        ~GraphBuilder() = default;

        virtual void initialize(std::vector<float> multiplier) = 0;

        virtual NeighborListShapes get_neighbor_list_shapes(
            int max_atoms, int inum, int* numneigh) = 0;

        virtual std::vector<xla::PjRtBuffer*> build_graph(
            xla::PjRtClient* client, int device_id, int inum, int *ilist,
                int *numneigh, int **firstneigh, bool update) = 0;
    };

    class SimpleSparseNeighborList : public GraphBuilder {

        public:
            // TODO: Documentation
            //
            // @param ilist: Index of the sender atom
            // @param numneigh: Number of neighbors for the sender atom
            // @param firstneigh: Start of the neighbor list for the first atom

            void initialize(std::vector<float> multipliers) override;

            NeighborListShapes get_neighbor_list_shapes(
                int max_atoms, int inum, int* numneigh) override;

            std::vector<xla::PjRtBuffer*> build_graph(
                xla::PjRtClient* client, int device_id, int inum, int *ilist,
                int *numneigh, int **firstneigh, bool update) override;

        private:
            float edge_multiplier;

            std::unique_ptr<xla::Literal> senders_literal;
            std::unique_ptr<xla::Literal> receivers_literal;

            std::unique_ptr<xla::PjRtBuffer> senders_buffer;
            std::unique_ptr<xla::PjRtBuffer> receivers_buffer;

            int n_edges = 0;
            int fill_value = 0;

    };

} // namespace jcn

#endif //GRAPH_BUILDER_H
