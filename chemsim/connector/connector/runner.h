//
// Created by Paul Fuchs on 24.09.24.
//

#include "connector/compiler.h"
#include "connector/graph_builder.h"
#include "connector/libconnector.h"
#include "connector/domain.h"
#include "connector/model.pb.h"

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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

    class Runner {
    public:
        Runner(ConnectorConfig config);
        ~Runner() = default;

        // Computes the forces and writes them directly to the force array
        double compute_forces(
            int inum, int gnum, double **x, double** f, int *type, int *ilist,
            int *numneigh, int **firstneigh, bool list_changed);

        ModelProperties get_model_properties();

    private:
        std::unique_ptr<chemsim::Model> model;

        std::unique_ptr<xla::PjRtClient> client;
        std::unique_ptr<xla::PjRtLoadedExecutable> executable;

        std::unique_ptr<GraphBuilder> neighbor_list;

        std::unique_ptr<AtomBuilder> atom_builder;
        std::unique_ptr<Compiler> compiler;

        ConnectorConfig config;

    };

} // namespace jcn

#endif //RUNNER_H
