#include "algorithms/eris/common.pb.h"
#include "algorithms/eris/config.h"
#include "algorithms/eris/coordinator.pb.h"
#include "mock_client.h"
#include "zmq.h"
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <future>
#include <gtest/gtest.h>
#include <random>
#include <string>

class ClientJoinTest : public testing::Test {
protected:
  ClientJoinTest(void) : client_{}, rng_(time(NULL)), dist_(1, 100), state_{} {}
  ~ClientJoinTest(void) = default;

  void SetUp(void) {
    eris::TrainingOptions options;
    uint32_t splits = dist_(rng_);

    options.set_min_clients(dist_(rng_));
    options.set_rounds(dist_(rng_));
    options.set_split_seed(dist_(rng_));
    options.set_splits(splits);
    *state_.mutable_options() = options;

    for (uint32_t i = 0; i < splits - 1; ++i) {
      eris::FragmentInfo fragment;
      fragment.set_id(i);
      fragment.set_submit_address("tcp://127.0.0.1:" +
                                  std::to_string(DEFAULT_ERIS_ROUTER_PORT + i));
      fragment.set_publish_address(
          "tcp://127.0.0.1:" + std::to_string(DEFAULT_ERIS_PUBLISH_PORT + i));

      *state_.add_aggregators() = fragment;
    }
    client_.set_parameters(std::vector<float>(100));
  }

  void TearDown() {
    EXPECT_TRUE(client_.get_dealer().is_empty());
    EXPECT_TRUE(client_.get_subscriber().is_empty());
  }

  std::future<void> AddResponse(eris::StateResponse &res) {
    zmq_msg_t msg;

    zmq_msg_init_size(&msg, res.ByteSizeLong());
    res.SerializeToArray(zmq_msg_data(&msg), res.ByteSizeLong());
    return client_.get_dealer().recv_enqueue(std::move(msg));
  }

  eris::StateRequest GetRequest(void) {
    zmq_msg_t msg;
    eris::StateRequest res;

    zmq_msg_init(&msg);
    EXPECT_TRUE(client_.get_dealer().send_dequeue(&msg));
    EXPECT_TRUE(res.ParseFromArray(zmq_msg_data(&msg), zmq_msg_size(&msg)));
    zmq_msg_close(&msg);

    return res;
  }

  void CheckAggregators(eris::State state) {
    for (int i = 0; i < int(state.options().splits()); ++i)
      if (i < state.aggregators().size()) {
        EXPECT_NE(client_.get_submit_sockets()[i], nullptr);
        EXPECT_NE(client_.get_publish_sockets()[i], nullptr);
        EXPECT_TRUE(client_.get_publish_sockets()[i]->subscribed());
      } else {
        EXPECT_EQ(client_.get_submit_sockets()[i], nullptr);
        EXPECT_EQ(client_.get_publish_sockets()[i], nullptr);
      }
  }

  void CheckJoinResponse(const eris::StateResponse &res) {
    EXPECT_TRUE(res.has_state());
    EXPECT_FALSE(res.has_error());

    const eris::State &state = res.state();
    EXPECT_TRUE(state.has_options());

    EXPECT_EQ(state.options().min_clients(),
              client_.get_options().min_clients());
    EXPECT_EQ(state.options().splits(), client_.get_options().splits());
    EXPECT_EQ(state.options().split_seed(), client_.get_options().split_seed());
    EXPECT_EQ(state.options().rounds(), client_.get_options().rounds());

    EXPECT_EQ(state.options().splits(), client_.get_submit_sockets().size());
    EXPECT_EQ(state.options().splits(), client_.get_submit_sockets().size());
    CheckAggregators(state);
  }

  MockClient client_;
  std::default_random_engine rng_;
  std::uniform_int_distribution<uint32_t> dist_;
  eris::State state_;
};

TEST_F(ClientJoinTest, JoinClient) {
  eris::StateResponse res;

  *res.mutable_state() = state_;
  std::future<void> response_received = AddResponse(res);

  EXPECT_TRUE(client_.mock_join());

  eris::StateRequest req = GetRequest();
  EXPECT_TRUE(req.has_join());
  EXPECT_FALSE(req.join().has_publish_address());
  EXPECT_FALSE(req.join().has_submit_address());

  response_received.wait();
  CheckJoinResponse(res);
  EXPECT_FALSE(client_.is_aggregator());
}

TEST_F(ClientJoinTest, JoinClientFailed) {
  eris::StateResponse res;
  eris::Error error;

  error.set_code(eris::ErrorCode::INVALID_ARGUMENT);
  error.set_msg("joining is not allowed");
  *res.mutable_error() = error;

  std::future<void> response_received = AddResponse(res);

  EXPECT_FALSE(client_.mock_join());

  eris::StateRequest req = GetRequest();
  EXPECT_TRUE(req.has_join());
  EXPECT_FALSE(req.join().has_publish_address());
  EXPECT_FALSE(req.join().has_submit_address());

  response_received.wait();
  EXPECT_FALSE(client_.is_aggregator());
}

TEST_F(ClientJoinTest, JoinAggregator) {
  eris::StateResponse res;

  state_.set_assigned_fragment(0);
  *res.mutable_state() = state_;

  std::future<void> response_received = AddResponse(res);
  EXPECT_TRUE(client_.set_aggregator_config("127.0.0.1", 10, 20));
  EXPECT_TRUE(client_.mock_join());

  eris::StateRequest req = GetRequest();
  EXPECT_TRUE(req.has_join());
  EXPECT_TRUE(req.join().has_publish_address());
  EXPECT_TRUE(req.join().has_submit_address());
  EXPECT_EQ(req.join().submit_address(), "tcp://127.0.0.1:10");
  EXPECT_EQ(req.join().publish_address(), "tcp://127.0.0.1:20");

  response_received.wait();
  CheckJoinResponse(res);
  EXPECT_TRUE(client_.is_aggregator());
}

TEST_F(ClientJoinTest, JoinAggregatorNoFragmentAssigned) {
  eris::StateResponse res;

  *res.mutable_state() = state_;
  std::future<void> response_received = AddResponse(res);
  EXPECT_TRUE(client_.set_aggregator_config("127.0.0.1", 10, 20));
  EXPECT_TRUE(client_.mock_join());

  eris::StateRequest req = GetRequest();
  EXPECT_TRUE(req.has_join());
  EXPECT_TRUE(req.join().has_publish_address());
  EXPECT_TRUE(req.join().has_submit_address());
  EXPECT_EQ(req.join().submit_address(), "tcp://127.0.0.1:10");
  EXPECT_EQ(req.join().publish_address(), "tcp://127.0.0.1:20");

  response_received.wait();
  CheckJoinResponse(res);
  EXPECT_FALSE(client_.is_aggregator());
}

TEST_F(ClientJoinTest, JoinAggregatorFailed) {
  eris::StateResponse res;
  eris::Error error;

  error.set_code(eris::ErrorCode::INVALID_ARGUMENT);
  error.set_msg("joining is not allowed");
  *res.mutable_error() = error;

  std::future<void> response_received = AddResponse(res);
  EXPECT_TRUE(client_.set_aggregator_config("127.0.0.1", 10, 20));
  EXPECT_FALSE(client_.mock_join());

  eris::StateRequest req = GetRequest();
  EXPECT_TRUE(req.has_join());
  EXPECT_TRUE(req.join().has_publish_address());
  EXPECT_TRUE(req.join().has_submit_address());
  EXPECT_EQ(req.join().submit_address(), "tcp://127.0.0.1:10");
  EXPECT_EQ(req.join().publish_address(), "tcp://127.0.0.1:20");

  response_received.wait();
  EXPECT_FALSE(client_.is_aggregator());
}
