#pragma once

#include "algorithms/eris/builder.h"
#include "algorithms/eris/common.pb.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.pb.h"
#include "erisfl/coordinator.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/server_callback.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/sync_stream.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using eris::CoordinatorUpdate;
using eris::FragmentInfo;
using eris::InitialUpdate;
using eris::JoinRequest;
using eris::TrainingOptions;
using grpc::CallbackServerContext;
using grpc::Channel;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::Status;
using grpc::StatusCode;

class ErisCoordinator final : public Coordinator {

public:
  explicit ErisCoordinator(const ErisCoordinatorBuilder &builder);

  ~ErisCoordinator(void);

  void start(void) override;
  void stop(void) override;

  inline uint32_t get_listening_port(void) const { return listening_port_; }

private:
  class Aggregators {
  public:
    explicit Aggregators(size_t splits);

    void handle_join_request(const JoinRequest *req, InitialUpdate *update);
    void wait_update(void);

  private:
    std::vector<FragmentInfo> aggregators_;
    std::mutex mu_;
    std::condition_variable cv_;
  };

  class CoordinatorImpl : public eris::Coordinator::CallbackService {
  public:
    explicit CoordinatorImpl(const TrainingOptions &options);

    grpc::ServerWriteReactor<CoordinatorUpdate> *
    Join(CallbackServerContext *ctx, const JoinRequest *req) override;

  private:
    const TrainingOptions options_;
    Aggregators aggregators_;
  };

  std::unique_ptr<Server> server_;
  uint16_t listening_port_;
  CoordinatorImpl service_;
  bool started_;
  const std::string listening_address_;
};
