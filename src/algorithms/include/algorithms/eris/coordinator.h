#pragma once

#include <cstring>
#include <future>
#include <spdlog/spdlog.h>
#include <vector>
#include <zmq.h>

#include "algorithms/eris/common.pb.h"
#include "algorithms/eris/config.h"
#include "algorithms/eris/coordinator.pb.h"
#include "algorithms/eris/service.h"
#include "erisfl/coordinator.h"
#include "util/networking.h"

/**
 * The ErisCoordinator class implements the Coordinator interface for the eris
 * federated training algorithm. In particular, it registers new joining clients
 * or aggregators, and publishes updates about the new joining aggregators to
 * the training clients. A client can start the training or register as
 * aggregator via a ZeroMQ router socket, and the events are published over a
 * ZeroMQ publisher socket.
 */
template <class Socket = ZMQSocket>
class ErisCoordinator final : public Coordinator {
  typedef std::vector<eris::FragmentInfo> State;

public:
  /**
   * It constructs an ErisCoordinator object with the provided
   * configurations.
   * Upon construction, the process will start listening on the provided
   * socket addresses.
   *
   * @param config The builder class carrying all the configurations to build
   * an ErisCoordinator.
   */
  explicit ErisCoordinator(const ErisCoordinatorConfig &config)
      : service_{&config}, options_{config.get_options()},
        state_(config.get_options().splits()) {}

  /**
   * Deletes an instance of an ErisCoordinator object.
   */
  ~ErisCoordinator(void) noexcept = default;

  /**
   * Starts the coordinator process. In practice, it will start handling
   * client requests and publishing events about new aggregators joining the
   * training.
   *
   * @param started An optional promise which will complete once the
   * coordinator process starts listening for connections from the clients.
   */
  void
  start(std::optional<std::promise<void>> started = std::nullopt) override {
    service_.start([&](zmq_msg_t *identity,
                       zmq_msg_t *msg) { handle_state(identity, msg); },
                   std::move(started));
  }

  /**
   * Stops the coordinator process. In practice, it will stop serving client
   * requests and publishing events for new aggregators joining the training.
   */
  void stop(void) override { service_.stop(); }

private:
  /**
   * Handles a request coming on the router socket.
   *
   * @param identity The identity of the client socket.
   * @param msg The request message.
   */
  void handle_state(zmq_msg_t *identity, zmq_msg_t *msg) noexcept {
    eris::StateRequest req;
    eris::StateResponse res;

    req.ParseFromArray(zmq_msg_data(msg), zmq_msg_size(msg));

    if (req.has_join()) {
      if (!valid_join_request(req.join(), res)) {
        service_.route_msg(identity, res);
        return;
      }

      handle_join(req.join(), res.mutable_state());
    }

    for (State::size_type i = 0; i < state_.size(); ++i)
      if (!state_[i].submit_address().empty())
        *res.mutable_state()->add_aggregators() = state_[i];

    service_.route_msg(identity, res);
  }

  /**
   * Validates a join request. In case the request is not valid, it sets the
   * corresponding error in the given response.
   *
   * @param req The joining request.
   * @param res The response message.
   * @return It returns true if it the request is valid; otherwise it returns
   * false.
   */
  bool valid_join_request(const eris::JoinRequest &req,
                          eris::StateResponse &res) noexcept {
    if (req.has_submit_address() && !req.has_publish_address()) {
      res.mutable_error()->set_code(eris::INVALID_ARGUMENT);
      res.mutable_error()->set_msg("Missing model updates publishing address");
      return false;

    } else if (req.has_publish_address() && !req.has_submit_address()) {
      res.mutable_error()->set_code(eris::INVALID_ARGUMENT);
      res.mutable_error()->set_msg("Missing weight submission address");
      return false;

    } else if (req.has_submit_address() &&
               !valid_zmq_endpoint(req.submit_address())) {
      res.mutable_error()->set_code(eris::INVALID_ARGUMENT);
      res.mutable_error()->set_msg("A weight submission address must have the "
                                   "form tcp://<address>:<port>"
                                   "where address is a valid IPv4 address");
      return false;

    } else if (req.has_publish_address() &&
               !valid_zmq_endpoint(req.publish_address())) {
      res.mutable_error()->set_code(eris::INVALID_ARGUMENT);
      res.mutable_error()->set_msg(
          "A model updates publishing address must have the form "
          "tcp://<address>:<port> where address is a valid IPv4 address");
      return false;
    }

    return true;
  }

  /**
   * Handles a joining request coming on the router socket by populating the
   * state portion of the response.
   *
   * @param req The request message.
   * @param state The state portion of the response.
   */
  void handle_join(const eris::JoinRequest &req, eris::State *state) noexcept {
    *state->mutable_options() = options_;
    if (!req.has_submit_address())
      return;

    for (State::size_type i = 0; i < state_.size(); ++i)
      if (state_[i].submit_address().empty()) {
        state_[i].set_id(i);
        state_[i].set_submit_address(req.submit_address());
        state_[i].set_publish_address(req.publish_address());
        state->set_assigned_fragment(i);
        service_.publish_event(state_[i]);
        spdlog::info("new aggregator joined for fragment {}. submit "
                     "address: {}, publish address: {}",
                     i, req.submit_address(), req.publish_address());

        break;
      }
  }

  friend class ErisCoordinatorTest;

  ErisService<Socket> service_; /**< The eris service handling the
                                 communications */

  const eris::TrainingOptions options_; /**< The training configuration */
  State state_;                         /**< The set of joined aggregators */
};
