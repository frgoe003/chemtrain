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

        virtual bool evaluate_statistics(
            std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>& results
        ) { return true; }; // Returns success if not required

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

    class DeviceSparseNeighborList : public GraphBuilder {

        public:

            void initialize(std::vector<float> multipliers) override;

            NeighborListShapes get_neighbor_list_shapes(
                int max_atoms, int inum, int* numneigh) override;

            std::vector<xla::PjRtBuffer*> build_graph(
                xla::PjRtClient* client, int device_id, int inum, int *ilist,
                int *numneigh, int **firstneigh, bool update) override;

            // Adjusts capacities if necessary
            bool evaluate_statistics(
                std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>& results
            ) override;

        private:

            float edge_multiplier;
            float capacity_multiplier;

            // This will trigger an overflow during the first execution
            // and return better estimates
            int n_edges = 729;
            int n_cells_x = 10;
            int n_cells_y = 10;
            int n_cells_z = 10;
            int capacity = 23;

            std::unique_ptr<xla::Literal> xcells_lit;
            std::unique_ptr<xla::Literal> ycells_lit;
            std::unique_ptr<xla::Literal> zcells_lit;
            std::unique_ptr<xla::Literal> capacity_lit;
            std::unique_ptr<xla::Literal> senders_lit;

            std::unique_ptr<xla::PjRtBuffer> xcells_buffer;
            std::unique_ptr<xla::PjRtBuffer> ycells_buffer;
            std::unique_ptr<xla::PjRtBuffer> zcells_buffer;
            std::unique_ptr<xla::PjRtBuffer> capacity_buffer;
            std::unique_ptr<xla::PjRtBuffer> senders_buffer;

            bool adjust_dimension(std::unique_ptr<xla::Literal>& cells, int size, xla::PrimitiveType type);

    };

} // namespace jcn

#endif //GRAPH_BUILDER_H
