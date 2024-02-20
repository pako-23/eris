#pragma once

#include <grpcpp/grpcpp.h>

#include "algorithms/eris/coordinator.grpc.pb.h"
#include "erisfl/coordinator.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"

using grpc::CallbackServerContext;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::Status;

class ErisCoordinator final : public Coordinator {
 public:
  explicit ErisCoordinator(void);
  void run(void) override;

 private:
  class CoordinatorImpl final
      : public coordinator::Coordinator::CallbackService {
   public:
    explicit CoordinatorImpl(void);
    grpc::ServerUnaryReactor* Join(CallbackServerContext*,
                                   const coordinator::JoinRequest*,
                                   coordinator::JoinResponse*) override;
  };
};
