#pragma once

#include "algorithms/eris/builder.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.pb.h"
#include "erisfl/coordinator.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/support/status.h>
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
  explicit ErisCoordinator(const ErisCoordinatorBuilder &);

  void start(void) override;
  ~ErisCoordinator(void);

private:
  class ClientConnection final {
    explicit ClientConnection(std::shared_ptr<Channel>);
  };

  class CoordinatorImpl final
      : public coordinator::Coordinator::CallbackService {
  public:
    explicit CoordinatorImpl(const coordinator::TrainingOptions &);
    grpc::ServerUnaryReactor *Join(CallbackServerContext *,
                                   const coordinator::JoinRequest *,
                                   coordinator::JoinResponse *) override;

  private:
    const coordinator::TrainingOptions &options_;
    std::unordered_multimap<std::string, ClientConnection> clients_;
    std::vector<std::string> aggregators_;
  };

  const ErisCoordinatorBuilder &builder_;
};
