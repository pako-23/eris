#pragma once

#include "algorithms/eris/builder.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.pb.h"
#include "erisfl/coordinator.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/server_callback.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/sync_stream.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

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

private:
  class CoordinatorImpl final
      : public coordinator::Coordinator::CallbackService {
  public:
    explicit CoordinatorImpl(const coordinator::TrainingOptions &opt);

    grpc::ServerWriteReactor<coordinator::CoordinatorUpdate> *
    Join(CallbackServerContext *ctx,
         const coordinator::JoinRequest *req) override;

  private:
    const coordinator::TrainingOptions &options_;

    std::vector<std::string> aggregators_;
  };

  const ErisCoordinatorBuilder &builder_;
  std::unique_ptr<Server> server_;
};
