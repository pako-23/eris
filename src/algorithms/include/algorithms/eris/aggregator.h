#pragma once

#include "algorithms/eris/aggregator.grpc.pb.h"
#include "algorithms/eris/aggregator.pb.h"
#include "algorithms/eris/builder.h"
#include "algorithms/eris/common.pb.h"
#include "algorithms/eris/service.h"
#include <cstddef>
#include <cstdint>
#include <grpcpp/server.h>
#include <grpcpp/server_context.h>
#include <mutex>
#include <vector>

using eris::Empty;
using eris::FragmentWeights;
using eris::WeightUpdate;
using grpc::CallbackServerContext;
using grpc::Server;

/**
 * The ErisAggregator class aggreagtes the weights submitted during the training
 * phase by the clients to obtain an updated version of the model at each round
 * of the training. A client can submit his weights via a gRPC interface, and
 * the event publishing happens via a ZeroMQ interface.
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
  ~ErisAggregator(void) = default;

  /**
   * Starts the aggregation process. In practice, it will start serving RPC
   * requests and publishing events about model fragment changes.
   */
  void start(void) noexcept;

  /**
   * Stops the aggregation process. In practice, it will stop serving RPC
   * requests and publishing events.
   */
  void stop(void) noexcept;

  /**
   * It returns the port on which the aggregator is publishing events about
   * model fragment changes.
   *
   * @return The port on which the aggregator is publishing events about model
   * fragment changes.
   */
  inline uint16_t get_publish_port(void) const {
    return service_.get_publish_port();
  }

  /**
   * It returns the port on which the aggregator is listening for RPC requests.
   *
   * @return The port on which the aggregator is listening for RPC requests.
   */
  inline uint16_t get_rpc_port(void) const noexcept {
    return service_.get_rpc_port();
  }

private:
  /**
   * The AggregatorImpl class implements the Aggregator gRPC interface.
   */
  class AggregatorImpl : public eris::Aggregator::CallbackService {
  public:
    /**
     * It constructs an AggregatorImpl object with the provided training
     * configurations and the registering ErisAggregator.
     *
     * @param aggregator The ErisAggregator that registerd the Aggregator
     * service.
     * @param builder The builder carrying all the aggregation configurations.
     */
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
                                            Empty *res) override;

    grpc::ServerUnaryReactor *GetUpdate(CallbackServerContext *ctx,
                                        const Empty *req,
                                        WeightUpdate *res) override;

  private:
    std::mutex prev_mu_;
    WeightUpdate prev_update_;

    uint32_t round_; /**< The current round of the training */
    std::vector<float>
        weights_; /**< The accumulated weights shared by the clients */
    uint32_t contributors_; /**< The number of contributing clients */
    std::mutex
        mu_; /**< A mutex providing mulital exclusion the internal state */
    const size_t fragment_size_; /**< The size of the assigned fragment */
    const uint32_t min_clients_; /**< The minimum number of weight contributions
                                    required before sharing an update */
    ErisAggregator *aggregator_; /**< The registrating ErisAggregator */
  };

  ErisService<AggregatorImpl> service_; /**< The ErisService that will listen on
                   gRPC requests and publish events.*/
};
