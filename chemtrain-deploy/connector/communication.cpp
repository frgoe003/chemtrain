#include "connector/communication.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>

#include "absl/status/status.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/type_registry.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace jcn {

CommunicationWorkspace::CommunicationWorkspace()
    : worker_(&CommunicationWorkspace::WorkerLoop, this) {}

CommunicationWorkspace::~CommunicationWorkspace() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  ready_.notify_one();
  if (worker_.joinable()) worker_.join();
}

void CommunicationWorkspace::Schedule(
    stream_executor::StreamExecutor* executor, std::size_t bytes, Task task,
    Completion completion) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    jobs_.push(Job{executor, bytes, std::move(task), std::move(completion)});
  }
  ready_.notify_one();
}

void CommunicationWorkspace::WorkerLoop() {
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      ready_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
      if (stopping_ && jobs_.empty()) return;
      job = std::move(jobs_.front());
      jobs_.pop();
    }

    absl::Status status;
    try {
      if (job.executor == nullptr) {
        status = absl::InternalError(
            "communication job has no StreamExecutor");
      } else {
        if (buffer_ == nullptr || buffer_executor_ != job.executor ||
            buffer_capacity_ < job.bytes) {
          auto allocation = job.executor->HostMemoryAllocate(job.bytes);
          if (!allocation.ok()) {
            status = allocation.status();
          } else {
            buffer_ = std::move(allocation).value();
            buffer_executor_ = job.executor;
            buffer_capacity_ = job.bytes;
          }
        }
        if (status.ok()) status = job.task(buffer_->address().opaque());
      }
    } catch (const std::exception& error) {
      status = absl::InternalError(error.what());
    } catch (...) {
      status = absl::InternalError("unknown communication worker failure");
    }
    job.completion(std::move(status));
  }
}

CommunicationContext::CommunicationContext(CommunicationCallbacks callbacks,
                                           bool enabled,
                                           CommunicationWorkspace* workspace)
    : callbacks_(callbacks), enabled_(enabled), workspace_(workspace) {}

std::int64_t CommunicationContext::ActiveRows(std::int64_t capacity) const {
  if (!enabled_ || callbacks_.active_rows == nullptr) return capacity;
  // A/B switch for measuring whether active-prefix staging beats transferring
  // the complete static-capacity buffer on a specific system.
  if (std::getenv("JCN_COMM_STAGE_FULL_BUFFER") != nullptr) return capacity;
  const std::int64_t rows = callbacks_.active_rows(callbacks_.context);
  return rows >= 0 && rows <= capacity ? rows : capacity;
}

absl::Status CommunicationContext::Exchange(
    void* data, std::int64_t rows, std::int64_t cols,
    CommunicationScalarType type, bool reverse) {
  if (!enabled_) return absl::OkStatus();

  if (callbacks_.exchange == nullptr) {
    return absl::FailedPreconditionError(
        "communicating model executed without LAMMPS communication callbacks");
  }

  std::unique_lock<std::mutex> lock(mutex_);

  request_done_.wait(lock, [this] { return !pending_; });

  data_ = data;
  rows_ = rows;
  cols_ = cols;
  type_ = type;
  reverse_ = reverse;
  error_.clear();
  completed_ = false;
  pending_ = true;

  request_ready_.notify_one();
  request_done_.wait(lock, [this] { return completed_; });

  std::string error = error_;

  pending_ = false;
  completed_ = false;
  request_done_.notify_all();

  if (!error.empty()) return absl::InternalError(error);
  return absl::OkStatus();
}

bool CommunicationContext::ServiceOne() {
  void* data;
  std::int64_t rows;
  std::int64_t cols;
  CommunicationScalarType type;
  bool reverse;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pending_ || servicing_ || completed_) return false;

    servicing_ = true;
    data = data_;
    rows = rows_;
    cols = cols_;
    type = type_;
    reverse = reverse_;
  }

  const char* callback_error = nullptr;
  int rc = callbacks_.exchange(callbacks_.context, data, rows, cols, type,
                               reverse, &callback_error);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    error_.clear();

    if (rc != 0) {
      error_ = callback_error == nullptr
                   ? "LAMMPS communication callback failed"
                   : callback_error;
    }

    servicing_ = false;
    completed_ = true;
  }

  request_done_.notify_all();
  return true;
}

bool CommunicationContext::HasPending() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pending_ && !completed_;
}

namespace {

namespace ffi = xla::ffi;
namespace se = stream_executor;

bool CommunicationDebugEnabled() {
  static const bool enabled = std::getenv("JCN_COMM_DEBUG") != nullptr;
  return enabled;
}

xla::ffi::TypeRegistry::TypeId g_communication_context_type_id =
    xla::ffi::TypeRegistry::kUnknownTypeId;

tsl::AsyncValueRef<tsl::Chain> RunExchange(
    ffi::AnyBuffer input, ffi::Result<ffi::AnyBuffer> output,
    se::Stream* stream, CommunicationContext* context, bool reverse) {
  auto done = tsl::MakeConstructedAsyncValueRef<tsl::Chain>();

  auto fail = [&done](std::string message) {
    done.SetError(absl::InternalError(std::move(message)));
  };

  if (stream == nullptr) {
    fail("chemtrain communication FFI called without stream");
    return done;
  }

  if (context == nullptr) {
    fail("chemtrain communication FFI called without CommunicationContext");
    return done;
  }

  if (context->workspace() == nullptr) {
    fail("chemtrain communication FFI called without workspace");
    return done;
  }

  if (input.dimensions().size() != 2 || output->dimensions().size() != 2 ||
      input.dimensions()[0] != output->dimensions()[0] ||
      input.dimensions()[1] != output->dimensions()[1] ||
      input.element_type() != output->element_type()) {
    fail("chemtrain gather expects matching rank-2 input and output buffers");
    return done;
  }

  CommunicationScalarType scalar_type;
  if (input.element_type() == xla::F32) {
    scalar_type = CommunicationScalarType::F32;
  } else if (input.element_type() == xla::F64) {
    scalar_type = CommunicationScalarType::F64;
  } else {
    fail("chemtrain gather supports only f32 and f64 buffers");
    return done;
  }

  const std::int64_t rows = input.dimensions()[0];
  const std::int64_t cols = input.dimensions()[1];
  const std::size_t bytes = input.size_bytes();
  const std::int64_t active_rows = context->ActiveRows(rows);
  const std::size_t element_bytes =
      scalar_type == CommunicationScalarType::F32 ? sizeof(float)
                                                   : sizeof(double);
  const std::size_t active_bytes =
      static_cast<std::size_t>(active_rows) *
      static_cast<std::size_t>(cols) * element_bytes;

  void* input_data = input.untyped_data();
  void* output_data = output->untyped_data();
  CommunicationWorkspace* workspace = context->workspace();
  workspace->Schedule(
      stream->parent(), active_bytes,
      [=](void* host) -> absl::Status {
        se::DeviceAddressBase src(input_data, bytes);
        se::DeviceAddressBase dst(output_data, bytes);

        // Preserve the inactive padded tail on device and stage only atom rows
        // that LAMMPS can access. This trades one D2D copy for less PCIe data.
        absl::Status status = stream->Memcpy(&dst, src, bytes);
        if (!status.ok()) return status;

        if (!context->enabled()) return stream->BlockHostUntilDone();

        status = stream->Memcpy(host, src, active_bytes);
        if (!status.ok()) return status;
        status = stream->BlockHostUntilDone();
        if (!status.ok()) return status;

        status = context->Exchange(host, active_rows, cols, scalar_type,
                                   reverse);
        if (!status.ok()) return status;

        status = stream->Memcpy(&dst, host, active_bytes);
        if (!status.ok()) return status;
        return stream->BlockHostUntilDone();
      },
      [done](absl::Status status) mutable {
        if (status.ok()) {
          done.SetStateConcrete();
        } else {
          done.SetError(std::move(status));
        }
      });
  return done;
}

tsl::AsyncValueRef<tsl::Chain> GatherForward(
    ffi::AnyBuffer input, ffi::Result<ffi::AnyBuffer> output,
    se::Stream* stream, CommunicationContext* context) {
  return RunExchange(input, output, stream, context, false);
}

tsl::AsyncValueRef<tsl::Chain> GatherReverse(
    ffi::AnyBuffer input, ffi::Result<ffi::AnyBuffer> output,
    se::Stream* stream, CommunicationContext* context) {
  return RunExchange(input, output, stream, context, true);
}

XLA_FFI_DEFINE_HANDLER(
    kGatherForward, GatherForward,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::UserData<CommunicationContext>>());

XLA_FFI_DEFINE_HANDLER(
    kGatherReverse, GatherReverse,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::UserData<CommunicationContext>>());

const PJRT_FFI* FindFfiExtension(const PJRT_Api* api) {
  if (api == nullptr) return nullptr;

  for (PJRT_Extension_Base* ext = api->extension_start; ext != nullptr;
       ext = ext->next) {
    if (ext->type == PJRT_Extension_Type_FFI) {
      return reinterpret_cast<const PJRT_FFI*>(ext);
    }
  }

  return nullptr;
}

absl::Status RegisterCommunicationContextType(const PJRT_FFI* ffi) {
  if (g_communication_context_type_id !=
      xla::ffi::TypeRegistry::kUnknownTypeId) {
    return absl::OkStatus();
  }

  if (ffi == nullptr || ffi->type_register == nullptr) {
    return absl::InternalError(
        "PJRT FFI extension does not provide type_register");
  }

  // Critical:
  // This must match what ffi::UserData<CommunicationContext> asks for.
  // In this XLA revision, TypeRegistry::GetTypeName<T>() is typeid(T).name(),
  // not a demangled string such as "jcn::CommunicationContext".
  absl::string_view type_name =
      xla::ffi::TypeRegistry::GetTypeName<CommunicationContext>();

  // Match the C++ FFI type info used by internal::GetTypeId<T>(api).
  // If this does not match, XLA may assign or expect a different type id.
  xla::ffi::TypeRegistry::TypeInfo cpp_type_info =
      xla::ffi::TypeRegistry::GetTypeInfo<CommunicationContext>();

  PJRT_FFI_Type_Info type_info;
  std::memset(&type_info, 0, sizeof(type_info));
  type_info.deleter = cpp_type_info.deleter;
  type_info.serialize = nullptr;
  type_info.deserialize = nullptr;

  PJRT_FFI_Type_Register_Args args;
  std::memset(&args, 0, sizeof(args));

  args.struct_size = PJRT_FFI_Type_Register_Args_STRUCT_SIZE;
  args.extension_start = nullptr;  // Keep this only if your struct has it.
  args.type_name = type_name.data();
  args.type_name_size = type_name.size();
  args.type_id = 0;
  args.type_info = &type_info;

  PJRT_Error* error = ffi->type_register(&args);
  if (error != nullptr) {
    return absl::InternalError(
        "Failed to register CommunicationContext FFI type");
  }

  g_communication_context_type_id =
      xla::ffi::TypeRegistry::TypeId(args.type_id);

  if (g_communication_context_type_id ==
      xla::ffi::TypeRegistry::kUnknownTypeId) {
    return absl::InternalError(
        "PJRT FFI registered CommunicationContext with unknown type id");
  }

  if (CommunicationDebugEnabled()) {
    std::cerr << "Registered CommunicationContext FFI type name='"
              << std::string(type_name)
              << "' id="
              << g_communication_context_type_id.value()
              << std::endl;
  }

  return absl::OkStatus();
}

int RegisterOne(const PJRT_FFI* ffi, const char* name,
                XLA_FFI_Handler* handler,
                const char* platform_name) {
  PJRT_FFI_Register_Handler_Args args;
  std::memset(&args, 0, sizeof(args));

  args.struct_size = PJRT_FFI_Register_Handler_Args_STRUCT_SIZE;
  args.target_name = name;
  args.target_name_size = std::strlen(name);
  args.handler = reinterpret_cast<void*>(handler);
  args.platform_name = platform_name;
  args.platform_name_size = std::strlen(platform_name);
  args.traits = static_cast<PJRT_FFI_Handler_TraitsBits>(0);

  return ffi->register_handler(&args) == nullptr ? 0 : 1;
}

}  // namespace

absl::Status AddCommunicationContextToExecuteContext(
    xla::ExecuteContext* execute_context,
    CommunicationContext* communication_context) {
  if (execute_context == nullptr) {
    return absl::InvalidArgumentError("execute_context must not be null");
  }

  if (communication_context == nullptr) {
    return absl::InvalidArgumentError(
        "communication_context must not be null");
  }

  if (g_communication_context_type_id ==
      xla::ffi::TypeRegistry::kUnknownTypeId) {
    return absl::FailedPreconditionError(
        "CommunicationContext FFI type was not registered");
  }

  return execute_context->ffi_context().Insert(
      g_communication_context_type_id,
      communication_context);
}

int RegisterCommunicationFfi(const PJRT_Api* api, const char* platform_name) {
  if (CommunicationDebugEnabled()) {
    std::cerr << "RegisterCommunicationFfi called for " << platform_name
              << std::endl;
  }

  const PJRT_FFI* ffi = FindFfiExtension(api);
  if (ffi == nullptr || ffi->register_handler == nullptr) {
    std::cerr << "PJRT FFI extension/register_handler unavailable"
              << std::endl;
    return 1;
  }

  absl::Status type_status = RegisterCommunicationContextType(ffi);
  if (!type_status.ok()) {
    std::cerr << "RegisterCommunicationContextType failed: "
              << type_status.ToString() << std::endl;
    return 1;
  }

  int rc = 0;
  rc |= RegisterOne(ffi, "chemtrain_deploy.gather_forward", kGatherForward,
                    platform_name);
  rc |= RegisterOne(ffi, "chemtrain_deploy.gather_reverse", kGatherReverse,
                    platform_name);

  if (CommunicationDebugEnabled()) {
    std::cerr << "RegisterCommunicationFfi rc=" << rc << std::endl;
  }
  return rc;
}

}  // namespace jcn
