#ifndef main_h
#define main_h

#include <vector>
#include <memory>

//#include "xla/pjrt/pjrt_client.h"
//#include "xla/pjrt/pjrt_c_api_client.h"
//#include "xla/pjrt/pjrt_executable.h"
//#include "xla/pjrt/pjrt_stream_executor_client.h"
//
//#include "xla/literal.h"
//#include "xla/literal_util.h"

namespace jcn {

	void execute();

    class Connector {
        public:
            // Compute the force by evaluating a HLO module from JAX
            Connector();
            std::vector<std::vector<float>> force(std::vector<std::vector<float>> position, std::vector<std::vector<int>> neighbors);

          // Stores the compiled module
        private:
            class Connect;
            std::unique_ptr<Connect> connect_instance;
    };
};

#endif
