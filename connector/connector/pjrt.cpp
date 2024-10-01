//
// Created by Paul Fuchs on 01.10.24.
//

#include "pjrt.h"

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"

namespace jcn {

    std::unique_ptr<xla::PjRtBuffer> create_buffer(xla::PjRtClient* client, int device_id, xla::Literal* literal) {
        absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> input_buffer = client->BufferFromHostBuffer(
            literal->untyped_data(),
            literal->shape().element_type(),
            literal->shape().dimensions(),
            std::optional<absl::Span<int64_t const>>{},
            xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
            []() { /* frees literal */ },
            client->addressable_devices()[device_id]
        );

        if (!input_buffer.ok()) {
            throw std::runtime_error("Failed to create buffer: " + input_buffer.status().ToString());
        }

        return std::move(input_buffer).value();
    }

} // namespace jcn
