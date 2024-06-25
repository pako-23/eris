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
using grpc::Server;

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
   * Upon construction, the process will start listening on the provided publish
   * address and RPC address.
   *
   * @param builder The builder class carrying all the configurations to build
   * an ErisCoordinator.
   */
  explicit ErisCoordinator(const ErisCoordinatorBuilder &builder);

  /**
   * Deletes an instance of an ErisCoordinator object.
   */
  ~ErisCoordinator(void);

  /**
   * Starts the coordinator process. In practice, it will start serving RPC
   * requests and publishing events about new aggregators joining the training.
   */
  void start(void) override;

  /**
   * Stops the coordinator process. In practice, it will stop serving RPC
   * requests and publishing events for new aggregators joining the training.
   */
  void stop(void) override;

  /**
   * It returns the port on which the coordinator is publishing events about new
   * aggregators joining the training.
   *
   * @return The port on which the coordinator is publishing events about new
   * aggregators joining the training.
   */
  inline uint16_t get_publish_port(void) const { return publish_port_; }

  /**
   * It returns the port on which the coordinator is listening for RPC requests.
   *
   * @return The port on which the coordinator is listening for RPC requests.
   */
  inline uint16_t get_rpc_port(void) const { return rpc_port_; }

  /**
   * It publishes an event about a new aggregator joining.
   *
   * @param info The newly joined aggregator endpoints with the fragment
   * identifier that it got assigned.
   * @return It returns true if it successfully manges to publish the event
   * related to the new aggregator joining; otherwise it returns false.
   */
  bool publish_aggregator(const FragmentInfo &info);

private:
  /**
   * The CoordinatorImpl class implements the Coordinator RPC interface.
   */
  class CoordinatorImpl : public eris::Coordinator::CallbackService {
  public:
    /**
     * It constructs a CoordinatorImpl object with the provided training
     * configurations and the registering ErisCoordinator.
     *
     * @param options The configurations that should be used during the
     * training.
     * @param coordinator The ErisCoordinator that registerd the Coordinator
     * service.
     */
    explicit CoordinatorImpl(const TrainingOptions &options,
                             ErisCoordinator *coordinator);

    /**
     * It implements the Join functionality for an ErisCoordinator.
     *
     * @param ctx The gRPC server call context.
     * @param req The gRPC Join request containing information about the joining
     * client or aggregator.
     * @param res The gRPC Join response containing all the configurations for
     * the joining client or aggregator.
     */
    grpc::ServerUnaryReactor *Join(CallbackServerContext *ctx,
                                   const JoinRequest *req,
                                   InitialState *res) override;

  private:
    const TrainingOptions options_;         /**< The training configurations */
    std::vector<FragmentInfo> aggregators_; /**< The mapping from fragment ID to
                                               aggregator */
    std::mutex mu_; /**< A mutex providing mutual exclusion on aggregators_ */
    ErisCoordinator *coordinator_; /**< The registering ErisCoordinator */
  };

  std::unique_ptr<Server> server_; /**< The listening gRPC server */
  void *zmq_ctx;                   /**< The ZeroMQ socket context */
  void *publisher;                 /**< The ZeroMQ publisher socket  */
  uint16_t rpc_port_;              /**< The effective RPC listening port */
  uint16_t publish_port_;          /**< The effective publishing port */
  CoordinatorImpl service_;        /**< The implementation of the gRPC
                                        Coordinator interface */
  bool started_; /**< If the ErisCoordinator has been started */
  const std::string listening_address_; /**< The RPC listening address from the
                                           builder configurations */
};
