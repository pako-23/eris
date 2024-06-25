#pragma once

#include "algorithms/eris/aggregator.grpc.pb.h"
#include "algorithms/eris/aggregator.pb.h"
#include "algorithms/eris/builder.h"
#include "algorithms/eris/common.pb.h"
#include <cstddef>
#include <cstdint>
#include <grpcpp/server.h>
#include <grpcpp/server_context.h>
#include <mutex>
#include <vector>

using eris::FragmentWeights;
using eris::WeightUpdate;
using grpc::CallbackServerContext;
using grpc::Server;

/**
 * The ErisAggregator class aggreagtes the weights submitted during the training
 * phase by the clients to obtain an updated version of the model at each round
 * of the training.
 */
class ErisAggregator final {
public:
  /**
   * It constructs an ErisAggregator object with the provided configurations.
   * Upon construction, the process will start listening on the provided publish
   * address and RPC address.
   *
   * @param builder The builder class carrying all the configurations to build
   * an ErisAggregator.
   */
  explicit ErisAggregator(const ErisAggregatorBuilder &builder);

  /**
   * Deletes an instance of an ErisAggregator object.
   */
  ~ErisAggregator(void);

  /**
   * Starts the aggregation process. In practice, it will start serving RPC
   * requests and publishing events about model fragment changes.
   */
  void start(void);

  /**
   * Stops the aggregation process. In practice, it will stop serving RPC
   * requests and publishing events about model fragment changes.
   */
  void stop(void);

  /**
   * It returns the port on which the aggregator is publishing events about
   * model fragment changes.
   *
   * @return The port on which the aggregator is publishing events about model
   * fragment changes.
   */
  inline uint16_t get_publish_port(void) const { return publish_port_; }

  /**
   * It returns the port on which the aggregator is listening for RPC requests.
   *
   * @return The port on which the aggregator is listening for RPC requests.
   */
  inline uint16_t get_rpc_port(void) const noexcept { return rpc_port_; }

  bool publish_weight(const WeightUpdate &update);

private:
  /**
   * The AggregatorImpl class implements the Aggregator RPC interface.
   */
  class AggregatorImpl : public eris::Aggregator::CallbackService {
  public:
    explicit AggregatorImpl(ErisAggregator *aggregator,
                            const ErisAggregatorBuilder &builder) noexcept;

    /**
     * It implements the submission of a fragment of the model weights for an
     * ErisAggregator.
     *
     * @param ctx The gRPC server call context.
     * @param req The gRPC request containing the weights of model fragment
     * coming from a requesting client.
     * @param res It returns an empty reply. An OK status code indicates the
     * successful submission of the weights.
     * @return The reactor that will handle the request.
     */
    grpc::ServerUnaryReactor *SubmitWeights(CallbackServerContext *ctx,
                                            const FragmentWeights *req,
                                            eris::Empty *res) override;

  private:
    uint32_t round_;
    std::vector<double> weights_;
    uint32_t contributors_;
    std::mutex mu_;
    const size_t fragment_size_;
    const uint32_t min_clients_;
    ErisAggregator *aggregator_;
  };

  std::unique_ptr<Server> server_; /**< The listening gRPC server */
  void *zmq_ctx;                   /**< The ZeroMQ socket context */
  void *publisher;                 /**< The ZeroMQ publisher socket  */
  uint16_t rpc_port_;              /**< The effective RPC listening port */
  uint16_t publish_port_;          /**< The effective publishing port */
  AggregatorImpl service_;         /**< The implementation of the gRPC
                                      Aggregator interface */
  bool started_; /**< If the ErisAggregator has been started */
  const std::string listening_address_; /**< The RPC listening address from the
                                           builder configurations */
};
