#include "libconnector.h"
#include "runner.h"

#include <iostream>
#include <string>
#include <vector>
#include <memory>  // For std::unique_ptr
#include <dlfcn.h>

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

    Connector::~Connector() = default;

    class Connector::Impl {
    public:
        Impl(ConnectorConfig config) : runner(config) {};
        ~Impl() = default;

        Runner runner;

    };

    double Connector::compute_force(int inum, int gnum, double **x, double** f, int *type, int *ilist,
            int *numneigh, int **firstneigh) {

        return impl_->runner.compute_forces(
          inum, gnum, x, f, type, ilist, numneigh, firstneigh);
    }

    Connector::Connector(ConnectorConfig config) : impl_(std::make_unique<Impl>(config)) {};

} // namespace jcn