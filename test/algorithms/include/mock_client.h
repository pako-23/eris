#pragma once

#include <algorithms/eris/client.h>
#include <algorithms/eris/config.h>
#include <algorithms/eris/coordinator.pb.h>
#include <algorithms/eris/split.h>
#include <cstdint>
#include <gtest/gtest.h>
#include <mock_zmq_socket.h>
#include <string>
#include <utility>
#include <vector>

class MockClient final : public ErisClient<std::vector<float>, MockZMQSocket> {
public:
  explicit MockClient(const std::string &router, const std::string &subscribe)
      : ErisClient<std::vector<float>, MockZMQSocket>{router, subscribe},
        parameters_{} {}

  explicit MockClient(void)
      : MockClient{
            "tcp://127.0.0.1:" + std::to_string(DEFAULT_ERIS_ROUTER_PORT),
            "tcp://127.0.0.1:" + std::to_string(DEFAULT_ERIS_PUBLISH_PORT)} {}

  std::vector<float> get_parameters(void) { return parameters_; }

  void set_parameters(const std::vector<float> &parameters) {
    parameters_ = parameters;
  }

  fit_result fit(void) { return std::make_pair(get_parameters(), 1); }
  void evaluate(void) {}

  inline MockZMQSocket &get_dealer(void) { return dealer_; }
  inline MockZMQSocket &get_subscriber(void) { return subscriber_; }
  inline bool mock_join(void) { return join(); }
  inline void mock_listen_coordinator_updates(void) {
    listen_coordinator_updates();
  }
  inline bool mock_receive_weights(uint32_t *round,
                                   std::vector<float> &parameters) {
    return receive_weights(round, parameters);
  }
  inline bool mock_submit_weights(uint32_t round) {

    return submit_weights(round, std::make_pair(get_parameters(), 1));
  }
  inline const std::string &get_aggr_address(void) const noexcept {
    return aggr_address_;
  }
  inline uint16_t get_aggr_submit_port(void) const { return aggr_submit_port_; }
  inline uint16_t get_aggr_publish_port(void) const {
    return aggr_publish_port_;
  }
  inline eris::TrainingOptions &get_options(void) { return options_; }
  inline RandomSplit &get_splitter(void) { return splitter_; }
  inline std::vector<std::unique_ptr<MockZMQSocket>> &get_submit_sockets(void) {
    return submit_;
  }
  inline std::vector<std::unique_ptr<MockZMQSocket>> &
  get_publish_sockets(void) {
    return publish_;
  }
  inline const bool is_aggregator(void) const { return aggregator_ != nullptr; }

  inline void lock(void) { mu_.lock(); }
  inline void unlock(void) { mu_.unlock(); }
  inline void notify(void) { cv_.notify_one(); }

private:
  std::vector<float> parameters_;
};
