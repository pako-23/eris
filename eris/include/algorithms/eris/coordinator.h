#pragma once

#include <cstdint>
#include <grpcpp/grpcpp.h>
#include <grpcpp/support/status.h>
#include <memory>
#include <string>
#include <zmq.hpp>

#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.pb.h"
#include "erisfl/coordinator.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"

using grpc::CallbackServerContext;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::Status;
using grpc::StatusCode;

class ErisCoordinator final
    : public Coordinator,
      public std::enable_shared_from_this<ErisCoordinator> {
public:
  explicit ErisCoordinator(const coordinator::TrainingOptions &options,
                           const std::string &address = "0.0.0.0",
                           uint16_t rpc_port = 5051, uint16_t pub_port = 5555);
  void start(void) override;
  ~ErisCoordinator(void);

private:
  class CoordinatorImpl final
      : public coordinator::Coordinator::CallbackService {
  public:
    explicit CoordinatorImpl(std::shared_ptr<ErisCoordinator>);
    grpc::ServerUnaryReactor *Join(CallbackServerContext *,
                                   const coordinator::JoinRequest *,
                                   coordinator::JoinResponse *) override;

    std::shared_ptr<ErisCoordinator> coordinator_;
  };

  struct Aggregator {
    const std::string publish_address;
    const std::string submit_address;
    const std::string id;
    Aggregator(const coordinator::Aggregator &);
  };

  const std::string grpc_address_;
  const std::string zmq_listening_address_;
  const std::string zmq_publish_address_;
  const coordinator::TrainingOptions options_;
  zmq::context_t zmq_context_;
  zmq::socket_t zmq_socket_;
  std::vector<std::unique_ptr<Aggregator>> aggregators_;
};
