#pragma once

#include "algorithms/eris/builder.h"
#include "algorithms/eris/common.pb.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.pb.h"
#include "erisfl/coordinator.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
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

using eris::FragmentInfo;
using eris::InitialState;
using eris::JoinRequest;
using eris::TrainingOptions;
using grpc::CallbackServerContext;
using grpc::Channel;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::Status;
using grpc::StatusCode;

/**
 * The ErisCoordinator class implements the Coordinator interface for the eris
 * federated training algorithm. In particular, it registers the clients and the
 * aggregators via a gRPC interface, and notifies the clients about new
 * aggregators joining via a ZeroMQ interface.
 */
class ErisCoordinator final : public Coordinator {

public:
  /**
   * It constructs an ErisCoordinator object with the provided configurations.
   *
   * @param builder The builder class carrying all the configurations to build
   * an ErisCoordinator
   */
  explicit ErisCoordinator(const ErisCoordinatorBuilder &builder);

  ~ErisCoordinator(void);

  void start(void) override;
  void stop(void) override;

  inline uint16_t get_pubssub_port(void) const { return pubsub_port_; }
  inline uint16_t get_rpc_port(void) const { return rpc_port_; }

  bool publish_aggregator(const FragmentInfo &info);

private:
  class CoordinatorImpl : public eris::Coordinator::CallbackService {
  public:
    explicit CoordinatorImpl(const TrainingOptions &options,
                             ErisCoordinator *coordinator);

    grpc::ServerUnaryReactor *Join(CallbackServerContext *ctx,
                                   const JoinRequest *req, InitialState *res);

  private:
    const TrainingOptions options_;
    std::vector<FragmentInfo> aggregators_;
    std::mutex mu_;
    ErisCoordinator *coordinator_;
  };

  std::unique_ptr<Server> server_;
  void *zmq_ctx;
  void *publisher;
  uint16_t rpc_port_;
  uint16_t pubsub_port_;
  CoordinatorImpl service_;
  bool started_;
  const std::string listening_address_;
};
