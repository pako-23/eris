#pragma once

#include "algorithms/eris/aggregation_strategy.h"
#include "algorithms/eris/aggregator.h"
#include "algorithms/eris/aggregator.pb.h"
#include "algorithms/eris/common.pb.h"
#include "algorithms/eris/coordinator.pb.h"
#include "algorithms/eris/split.h"
#include "erisfl/client.h"
#include "spdlog/spdlog.h"
#include "util/networking.h"
#include "zmq.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

/**
 * The ErisClient class implements the Client interface for the eris
 * federated training algorithm. In particular, a client tries to register to an
 * ErisCoordinator as a client or as an ErisAggregator. If it registers as an
 * ErisAggregator, a different thread will be started to perform the weights
 * aggregation steps. A client is responsible for joining the training,
 * performing the training steps, sharing its model parameters with the
 * aggregators, and updating the weights based on the ones published by the
 * aggregators.
 */
template <class Socket = ZMQSocket> class ErisClient : public Client {
public:
  /**
   * It constructs an ErisClient that will contact an ErisCoordinator at the
   * given addresses. Upon construction, the process will start connect to the
   * given addresses.
   *
   * @param router_address The address on which the ErisCoordinator will accept
   * requests.
   * @param subscribe_address The address on which the ErisCoordinator will
   * publish updates.
   */
  explicit ErisClient(const std::string &router_address,
                      const std::string &subscribe_address)
      : dealer_{ZMQ_DEALER}, subscriber_{ZMQ_SUB}, running_{false},
        aggr_address_{}, aggr_submit_port_{0}, aggr_publish_port_{0},
        aggr_strategy_{nullptr}, options_{}, splitter_{}, mu_{}, cv_{},
        submit_{}, publish_{}, coordinator_updates_{}, aggregator_{nullptr},
        aggregator_thread_{} {
    const int timeout = 100;

    if (!valid_zmq_endpoint(router_address))
      throw std::invalid_argument{
          "invalid endpoint address for coordinator router socket"};
    if (!valid_zmq_endpoint(subscribe_address))
      throw std::invalid_argument{
          "invalid endpoint address for coordinator publish socket"};

    if (!dealer_.connect(router_address))
      throw std::runtime_error{
          "failed to connect to given coordinator router address"};

    if (!subscriber_.connect(subscribe_address))
      throw std::runtime_error{
          "failed to connect to given coordinator publish address"};

    subscriber_.setsockopt(ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
    subscriber_.setsockopt(ZMQ_SUBSCRIBE, "", 0);
  }

  /**
   * Deletes an instance of an ErisClient object.
   */
  ~ErisClient(void) noexcept {
    if (aggregator_) {
      aggregator_->stop();
      aggregator_thread_.join();
    }
    running_ = false;
    if (coordinator_updates_.joinable())
      coordinator_updates_.join();
  }

  /**
   * Joins the training and performs the whole training process.
   *
   * @return If the training is successful, it returns true; otherwise, it
   * returns false.
   */
  bool train(void) override {
    uint32_t round = 0;
    if (!running_)
      return false;

    while (round != options_.rounds()) {
      std::pair<std::vector<float>, uint32_t> result = fit();

      if (!submit_weights(round, result) || !receive_weights(&round))
        return false;

      evaluate();
      spdlog::info("finished with round {0}", round);
      ++round;
    }

    return true;
  }

  /**
   * Configures the aggregator parameters. This method should be called before
   * the training starts; otherwise, it will have no effect.
   *
   * @return If the training is successful, it returns true; otherwise, it
   * returns false.
   */
  bool set_aggregator_config(const std::string &address,
                             uint16_t submit_port = 0,
                             uint16_t publish_port = 0) noexcept {
    if (!valid_ipv4(address) || address == "0.0.0.0" ||
        (submit_port == publish_port && submit_port != 0)) {
      return false;
    }

    aggr_address_ = address;
    aggr_submit_port_ = submit_port;
    aggr_publish_port_ = publish_port;
    return true;
  }

  void set_aggregation_strategy(
      std::shared_ptr<AggregationStrategy> strategy) noexcept {
    aggr_strategy_ = std::move(strategy);
  }

  /**
   * Returns the splitter used to split weights.
   *
   * @return The splitter used to split weights at each round.
   */
  inline const RandomSplit &get_splitter(void) const noexcept {
    return splitter_;
  }

  /**
   * It tries to join the training process. It will try to obtain the
   * configurations from the configured ErisCoordinator, and it will perform the
   * necessary configurations based on the response from the coordinator.
   *
   * @return If it manages to join the training process, it returns true;
   * otherwise, it returns false.
   */
  bool join(void) noexcept {
    eris::StateRequest req;
    eris::JoinRequest join_req;
    eris::StateResponse res;
    zmq_msg_t msg, reply;

    if (!aggr_address_.empty()) {
      setup_aggregator();

      aggregator_->get_publisher();
      join_req.set_submit_address(aggregator_->get_router().get_endpoint());
      join_req.set_publish_address(aggregator_->get_publisher().get_endpoint());
    }

    *req.mutable_join() = join_req;

    zmq_msg_init_size(&msg, req.ByteSizeLong());
    req.SerializeToArray(zmq_msg_data(&msg), req.ByteSizeLong());
    if (!dealer_.send_msg(&msg, 0)) {
      spdlog::error("failed to send join request to given coordinator");
      goto joining_failed;
    }

    zmq_msg_init(&reply);
    if (!dealer_.recv_msg(&reply, 0) ||
        !res.ParseFromArray(zmq_msg_data(&reply), zmq_msg_size(&reply))) {
      spdlog::error("failed to receive join response from the coordinator");
      goto joining_failed;
    }
    zmq_msg_close(&reply);

    if (res.has_error()) {
      spdlog::error("failed to join client with error: {}", res.error().msg());
      goto joining_failed;
    } else if (!res.has_state() && !res.state().has_options()) {
      spdlog::error("invalid state returned from the coordinator");
      goto joining_failed;
    }
    options_ = res.state().options();
    splitter_.configure(get_parameters().size(), options_.splits(),
                        options_.split_seed());

    if (res.state().has_assigned_fragment())
      start_aggregator(res.state().assigned_fragment());
    else
      aggregator_ = nullptr;

    submit_.resize(options_.splits());
    publish_.resize(options_.splits());

    for (const auto &aggr : res.state().aggregators())
      if (!register_aggregator(aggr)) {
        spdlog::error("failed to register aggregator with submit address {} "
                      "and publish address{}",
                      aggr.submit_address(), aggr.submit_address());
        goto joining_failed;
      }
    listen_coordinator_updates();
    running_ = true;
    return true;

  joining_failed:
    aggregator_ = nullptr;
    return false;
  }

private:
  /**
   * It tries to register a new aggregator for a specific fragment. It does no
   * peform any locking of the client state.
   *
   * @param aggr The aggregator to register.
   * @return If it manages to register the given aggregator, it returns true;
   * otherwise, it returns false.
   */
  bool register_aggregator(const eris::FragmentInfo &aggr) noexcept {
    submit_[aggr.id()] = std::make_unique<Socket>(ZMQ_DEALER);
    if (!submit_[aggr.id()]->connect(aggr.submit_address()))
      return false;

    publish_[aggr.id()] = std::make_unique<Socket>(ZMQ_SUB);
    if (!publish_[aggr.id()]->connect(aggr.publish_address()))
      return true;
    publish_[aggr.id()]->setsockopt(ZMQ_SUBSCRIBE, "", 0);

    return true;
  }

  /**
   * It tries to register a new aggregator for a specific fragment. It locks the
   * state before registering the new aggregator.
   *
   * @param aggr The aggregator to register.
   * @return If it manages to register the given aggregator, it returns true;
   * otherwise, it returns false.
   */
  bool register_aggregator_locked(const eris::FragmentInfo &aggr) noexcept {
    std::lock_guard lk(mu_);
    return register_aggregator(aggr);
  }

  /**
   * It starts a thread to listen on updates from the coordinator.
   */
  void listen_coordinator_updates(void) noexcept {
    coordinator_updates_ = std::thread{[&]() {
      zmq_msg_t msg;
      eris::FragmentInfo aggregator;

      zmq_msg_init(&msg);
      while (running_) {
        if (!subscriber_.recv_msg(&msg, 0))
          continue;

        if (!aggregator.ParseFromArray(zmq_msg_data(&msg),
                                       zmq_msg_size(&msg)) ||
            !register_aggregator_locked(aggregator)) {
          zmq_msg_close(&msg);
          zmq_msg_init(&msg);
          spdlog::error("failed to register received aggregator");
          continue;
        }

        cv_.notify_one();
        zmq_msg_close(&msg);
        zmq_msg_init(&msg);
      }
    }};
  }

  /**
   * It starts the aggregator thread for a given fragment.
   *
   * @param fragment_id The identifier of the fragment assigned to the
   * aggregator.
   */
  void start_aggregator(uint16_t fragment_id) noexcept {
    std::promise<void> started;
    std::future<void> started_ready = started.get_future();

    aggregator_->configure(splitter_.get_fragment_size(fragment_id),
                           options_.min_clients());

    aggregator_thread_ = std::thread(
        [](std::shared_ptr<ErisAggregator<Socket>> aggregator,
           std::promise<void> started) {
          aggregator->start(std::move(started));
        },
        aggregator_, std::move(started));

    started_ready.wait();
  }

  /**
   * It setups the aggregator. In practice, it will start the ErisService for
   * the aggregator, but it will not serve any request.
   */
  void setup_aggregator(void) noexcept {
    ErisServiceConfig config;
    config.set_router_address(aggr_address_);
    config.set_router_port(aggr_submit_port_);
    config.set_publish_address(aggr_address_);
    config.set_publish_port(aggr_publish_port_);

    if (aggr_strategy_)
      aggregator_ = std::make_shared<ErisAggregator<Socket>>(
          config, std::move(aggr_strategy_));
    else
      aggregator_ = std::make_shared<ErisAggregator<Socket>>(
          config, std::make_shared<WeightedAverage>());
  }

  /**
   * It submits the model parameters to the aggregators for a given training
   * round.
   *
   * @param round The current training round.
   * @return If it manages to succesfully submit the weights, it returns true;
   * otherwise, it returns false.
   */
  bool submit_weights(uint32_t round, const fit_result &parameters) noexcept {
    eris::WeightSubmissionResponse res;
    std::vector<eris::WeightSubmissionRequest> fragments =
        splitter_.split(parameters, round);

    for (size_t i = 0; i < fragments.size(); ++i) {
      zmq_msg_t msg, reply;
      bool submitted, received;

      {
        std::unique_lock lk(mu_);
        if (!submit_[i])
          cv_.wait(lk, [this, &i]() { return submit_[i] != nullptr; });
      }

      zmq_msg_init_size(&msg, fragments[i].ByteSizeLong());
      zmq_msg_init(&reply);
      fragments[i].SerializeToArray(zmq_msg_data(&msg),
                                    fragments[i].ByteSizeLong());

      {
        std::lock_guard lk(mu_);
        submitted = submit_[i]->send_msg(&msg, 0);
      }

      if (!submitted) {
        spdlog::error("failed to submit weights to aggregator with ID {}", i);
        zmq_msg_close(&reply);
        return false;
      }

      {
        std::lock_guard lk(mu_);
        received = submit_[i]->recv_msg(&reply, 0);
      }

      if (!received) {
        spdlog::error("failed to receive successful weight submission "
                      "notification from aggregator with ID {}",
                      i);
        zmq_msg_close(&reply);
        return false;
      } else if (!res.ParseFromArray(zmq_msg_data(&reply),
                                     zmq_msg_size(&reply))) {
        spdlog::error("failed to parseweight submission "
                      "notification from aggregator with ID {}",
                      i);
        zmq_msg_close(&reply);
        return false;
      } else if (res.has_error()) {
        spdlog::error("aggregator {} returned: {}", i, res.error().msg());
        zmq_msg_close(&reply);
        return false;
      }
      zmq_msg_close(&reply);
    }

    return true;
  }

  /**
   * It receives the weights from teh aggregators for a given training round.
   * Also, the round will be updates if the client receives weights for a higher
   * round. This could happen in case the client is some rounds behind with the
   * training.
   *
   * @param round The current training round. It will be updates if the client
   * receives weights from a higher round.
   * @return If it manages to succesfully receive the model parameters from all
   * the aggregators, it returns true; otherwise, it returns false.
   */
  bool receive_weights(uint32_t *round) noexcept {
    std::vector<eris::WeightUpdate> weights(options_.splits());
    std::vector<bool> done(options_.splits(), false);
    size_t i = 0;

    while (i < done.size()) {
      zmq_msg_t msg;
      bool received;

      if (done[i]) {
        ++i;
        continue;
      }

      weights[i] = eris::WeightUpdate{};
      zmq_msg_init(&msg);
      {
        std::lock_guard lk(mu_);
        received = publish_[i]->recv_msg(&msg, 0);
      }

      if (!received) {
        spdlog::error(
            "failed to receive weight updates from aggregator with ID {}", i);
        zmq_msg_close(&msg);
        return false;
      } else if (!weights[i].ParseFromArray(zmq_msg_data(&msg),
                                            zmq_msg_size(&msg))) {
        spdlog::error(
            "failed to deserialize weight updates from aggregator with ID {}",
            i);
        zmq_msg_close(&msg);
        return false;
      }

      zmq_msg_close(&msg);

      if (weights[i].round() == *round) {
        done[i++] = true;
      } else if (weights[i].round() > *round) {
        *round = weights[i].round();
        std::fill(done.begin(), done.end(), false);
        done[i] = true;
        i = 0;
      }
    }
    set_parameters(splitter_.reassemble(weights));

    return true;
  }

  friend class MockClient;

  Socket dealer_;            /**< The ZeroMQ socket sending requests to the
                                ErisCoordinator */
  Socket subscriber_;        /**< The ZeroMQ socket receiving updates from the
                                ErisCoordinator */
  std::atomic_bool running_; /**< Whether the process is running */

  std::string aggr_address_;   /**< The IPv4 address on which the aggregator
                                  should be listening to */
  uint16_t aggr_submit_port_;  /**< The port number on which the aggregator
                                  should be receiving weights submissions */
  uint16_t aggr_publish_port_; /**< The port number on which the aggregator
                                 should publish weight updates */
  std::shared_ptr<AggregationStrategy> aggr_strategy_; /**< The strategy used
                        for aggregating weights from a client */

  eris::TrainingOptions options_; /**< The training configurations coming from
                                     the ErisCoordinator */
  RandomSplit splitter_;       /**< The splitter splitting the model weights */
  std::mutex mu_;              /**< A mutex to provide mutual exclusion to the
                                  client's state */
  std::condition_variable cv_; /**< A condition variable notifying for updates
                                  in the list of known aggregators */
  std::vector<std::unique_ptr<Socket>> submit_;  /**< The list aggregator
                                                    submissions connections */
  std::vector<std::unique_ptr<Socket>> publish_; /**< The list aggregator
                                                    subscriptions */
  std::thread coordinator_updates_; /**< The thread listening for updates from
                                       the ErisCoordinator */

  std::shared_ptr<ErisAggregator<Socket>> aggregator_; /**< The
                                                          ErisAggregator
                                                          service */
  std::thread aggregator_thread_; /**< The thread running the
                                     aggregation process */
};
