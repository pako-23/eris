#pragma once

#include "algorithms/eris/aggregation_strategy.h"
#include "algorithms/eris/aggregator.pb.h"
#include "algorithms/eris/common.pb.h"
#include "algorithms/eris/config.h"
#include "algorithms/eris/service.h"
#include "spdlog/spdlog.h"
#include "zmq.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>

/**
 * The ErisAggregator class aggreagtes the weights submitted during the training
 * phase by the clients to obtain an updated version of the model at each round
 * of the training. A client can submit his weights over a ZeroMQ router socket
 * interface, and the event publishing happens via a ZeroMQ publish socket.
 */
template <class Socket = ZMQSocket> class ErisAggregator final {
public:
  /**
   * It constructs an ErisAggregator object with the provided configurations.
   * Upon construction, the process will start listening on the provided
   * addresses.
   *
   * @param config The configuration used to build the ErisAggregator.
   */
  explicit ErisAggregator(const ErisServiceConfig &config) noexcept
      : service_{&config}, aggregation_{std::make_unique<WeightedAverage>()},
        round_{0}, weights_{}, fragment_size_{0}, min_clients_{0} {}

  /**
   * Deletes an instance of an ErisAggregator object.
   */
  ~ErisAggregator(void) noexcept = default;

  /**
   * It configures the ErisAggregator. This method must be called before calling
   * the start method.
   *
   * @param fragment_size The size of the model fragment.
   * @param min_clients The minimum number of clients that should contribute
   * with their local weights before the ErisAggregator can publish a new model
   * weight update.
   */
  void configure(uint32_t fragment_size, uint32_t min_clients) noexcept {
    weights_.reserve(min_clients);
    fragment_size_ = fragment_size;
    min_clients_ = min_clients;
  }

  /**
   * Starts the aggregation process. In practice, it will start handling
   * the submissions of new model weights from the clients, and publising
   * events about new model updates.
   *
   * @param started An optional promise which will complete once the
   * aggregator process starts serving requests.
   */
  void
  start(std::optional<std::promise<void>> started = std::nullopt) noexcept {
    service_.start([&](zmq_msg_t *identity,
                       zmq_msg_t *msg) { handle_submission(identity, msg); },
                   std::move(started));
  }

  /**
   * Stops the aggregation process. In practice, it will stop handling
   * the submissions of new model weights from the clients, and publising events
   * about new model updates.
   */
  void stop(void) { service_.stop(); }

  inline Socket &get_publisher(void) { return service_.get_publisher(); }
  inline Socket &get_router(void) { return service_.get_router(); }

private:
  /**
   * Handles the submission of new model weights from a client.
   *
   * @param identity The identity of the client socket.
   * @param req The msg message containing the model weights from the client.
   */
  void handle_submission(zmq_msg_t *identity, zmq_msg_t *msg) noexcept {
    eris::WeightSubmissionRequest req;
    eris::WeightSubmissionResponse res;

    req.ParseFromArray(zmq_msg_data(msg), zmq_msg_size(msg));

    if ((size_t)req.weight_size() != fragment_size_) {
      res.mutable_error()->set_code(eris::INVALID_ARGUMENT);
      res.mutable_error()->set_msg("Wrong fragment size");
      service_.route_msg(identity, res);
      return;
    } else if (req.round() != round_) {
      res.mutable_error()->set_code(eris::INVALID_ARGUMENT);
      res.mutable_error()->set_msg("Wrong round number");
      service_.route_msg(identity, res);
      return;
    }

    weights_.emplace_back(req);

    service_.route_msg(identity, res);
    publish_model();
  }

  /**
   * It increases the number of contributors to build a new model. If the number
   * contributors reaches the minium number of clients required, it also
   * publishes the new model weights.
   */
  void publish_model(void) noexcept {
    eris::WeightUpdate update;

    if (weights_.size() < min_clients_)
      return;

    update = aggregation_->aggregate(round_, weights_);
    service_.publish_event(update);
    weights_.clear();
    ++round_;
  }

  friend class ErisAggregatorTest;

  ErisService<Socket> service_; /**< The eris service handling the
                                  communications */

  std::unique_ptr<AggregationStrategy> aggregation_; /**< The strategy used for
                                                        aggregating weights
                                                        received from clients */

  uint32_t round_; /**< The current round of the training */
  std::vector<eris::WeightSubmissionRequest> weights_; /**< The weights sent by
                                                          the clients */

  size_t fragment_size_; /**< The size of the assigned fragment */
  uint32_t min_clients_; /**< The minimum number of weight contributions
                                 required before sharing an update */
};
