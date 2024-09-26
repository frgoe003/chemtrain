//
// Created by Paul Fuchs on 24.09.24.
//

#include "compiler.h"
#include "graph_builder.h"
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

#ifndef RUNNER_H
#define RUNNER_H

namespace jcn {

    // Loads plugins and initializes the runtime
    void initialize();

    struct Atoms {
        // This information is required to refine the shapes of the energy
        // model
        int n_atoms;

        // The atom constructor keeps track wheter a recompilation of the
        // mlir module is necessary
        bool reallocate;

        // These are pointers to the actual data
        std::unique_ptr<xla::Literal> positions;
        std::unique_ptr<xla::Literal> species;
        std::unique_ptr<xla::Literal> ghost_mask;

    };

    class AtomBuilder {
    public:
        AtomBuilder(float atom_multiplier) : max_atoms(0), atom_multiplier(atom_multiplier) {};
        ~AtomBuilder() = default;

        // Padds the atom data to reduce number of recompilations
        Atoms build_domain(int inum, int gnum, double **x, int *type);

        // Writes back the force to the original array and returns the potential
        double evaluate_domain(int inum, double **f, std::shared_ptr<xla::Literal> forces, std::shared_ptr<xla::Literal> potential);

    private:
      int max_atoms;

      float atom_multiplier;

    };


    class Runner {
    public:
        Runner(ConnectorConfig config);
        ~Runner() = default;

        // Computes the forces and writes them directly to the force array
        double compute_forces(
            int inum, int gnum, double **x, double** f, int *type, int *ilist,
            int *numneigh, int **firstneigh);

    private:
        std::unique_ptr<xla::PjRtClient> client;
        std::unique_ptr<xla::PjRtLoadedExecutable> executable;

        SimpleSparseNeighborList neighbor_list;
        AtomBuilder atom_builder;
        Compiler compiler;

        ConnectorConfig config;

    };

} // namespace jcn

#endif //RUNNER_H
