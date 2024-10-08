//
// Created by Paul Fuchs on 01.10.24.
//

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"

#ifndef PJRT_H
#define PJRT_H

namespace jcn {

    std::unique_ptr<xla::PjRtBuffer> create_buffer(
        xla::PjRtClient* client, int device_id, xla::Literal* literal);

} // namespace jcn



#endif //PJRT_H
