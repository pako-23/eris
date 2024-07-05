#include "algorithms/eris/client.h"
#include "algorithms/eris/client_test_helpers.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <grpcpp/server_context.h>
#include <grpcpp/support/server_callback.h>
#include <grpcpp/support/status.h>
#include <gtest/gtest.h>
#include <mutex>
#include <random>

static const size_t clients = 8;
static const uint32_t min_clients = 5;
static const uint32_t rounds = 10;
static const uint32_t split_seed = 42;
static const uint32_t splits = 10;

class ClientJoinTest : public testing::Test, public ErisMockClient {
protected:
  ClientJoinTest(void)
      : coordinator_{min_clients, rounds, split_seed, splits}, rng(time(NULL)),
        dist(0, splits - 4) {}
  ~ClientJoinTest(void) {}

  void check_aggregators(ClientState &state) {
    for (size_t i = 0; i < state.get_subscriptions().size(); ++i)
      if (!coordinator_.get_aggregator(i)) {
        EXPECT_EQ(state.get_subscriptions()[i], nullptr);
        EXPECT_EQ(state.get_submitters()[i], nullptr);
      } else {
        EXPECT_NE(state.get_subscriptions()[i], nullptr);
        EXPECT_NE(state.get_submitters()[i], nullptr);
      }
  }

  void check_join(ClientState &state) {
    EXPECT_EQ(state.get_options().min_clients(),
              coordinator_.get_options().min_clients());
    EXPECT_EQ(state.get_options().rounds(),
              coordinator_.get_options().rounds());
    EXPECT_EQ(state.get_options().split_seed(),
              coordinator_.get_options().split_seed());
    EXPECT_EQ(state.get_options().splits(),
              coordinator_.get_options().splits());
    {
      std::lock_guard<ClientState> lk(state);
      EXPECT_EQ(state.get_submitters().size(), state.get_options().splits());
      EXPECT_EQ(state.get_subscriptions().size(), state.get_options().splits());
      check_aggregators(state);
    }
  }

  MockCoordinator coordinator_;
  std::array<ClientState, clients> states;

  std::default_random_engine rng;
  std::uniform_int_distribution<uint32_t> dist;
};

TEST_F(ClientJoinTest, JoinClient) {
  uint32_t size = dist(rng);

  for (uint32_t i = 0; i < size; ++i)
    coordinator_.add_aggregator(i, "127.0.0.1:50051", "tcp://127.0.0.1:5555");

  for (size_t i = 0; i < clients; ++i) {
    EXPECT_TRUE(states[i].join(this, coordinator_.get_rpc_address(),
                               coordinator_.get_pubsub_address()));
    check_join(states[i]);
  }
}

TEST_F(ClientJoinTest, JoinClientFailed) {
  coordinator_.set_fail_requests(true);
  EXPECT_FALSE(states[0].join(this, coordinator_.get_rpc_address(),
                              coordinator_.get_pubsub_address()));
}

TEST_F(ClientJoinTest, JoinAggregator) {
  uint32_t size = dist(rng);
  std::string aggr_address = "127.0.0.1";
  uint16_t aggr_rpc_port = 8080;
  uint16_t aggr_publish_port = 8081;

  for (uint32_t i = 0; i < size; ++i)
    coordinator_.add_aggregator(i, "127.0.0.1:50051", "tcp://127.0.0.1:5555");

  for (size_t i = 1; i < clients; ++i) {
    EXPECT_TRUE(states[i].join(this, coordinator_.get_rpc_address(),
                               coordinator_.get_pubsub_address()));
    check_join(states[i]);
  }

  EXPECT_TRUE(states[0].join(this, coordinator_.get_rpc_address(),
                             coordinator_.get_pubsub_address(), &aggr_address,
                             &aggr_rpc_port, &aggr_publish_port));
  check_join(states[0]);
  EXPECT_TRUE(states[0].is_aggregator());

  for (size_t i = 1; i < clients; ++i)
    check_aggregators(states[i]);
}

TEST_F(ClientJoinTest, JoinAggregatorNoFragmentAssigned) {
  std::string aggr_address = "127.0.0.1";
  uint16_t aggr_rpc_port = 8080;
  uint16_t aggr_publish_port = 8081;

  for (uint32_t i = 0; i < coordinator_.get_options().splits(); ++i)
    coordinator_.add_aggregator(i, "127.0.0.1:50051", "tcp://127.0.0.1:5555");

  for (size_t i = 1; i < clients; ++i) {
    EXPECT_TRUE(states[i].join(this, coordinator_.get_rpc_address(),
                               coordinator_.get_pubsub_address()));
    check_join(states[i]);
  }

  EXPECT_TRUE(states[0].join(this, coordinator_.get_rpc_address(),
                             coordinator_.get_pubsub_address(), &aggr_address,
                             &aggr_rpc_port, &aggr_publish_port));
  check_join(states[0]);
  EXPECT_FALSE(states[0].is_aggregator());

  for (size_t i = 1; i < clients; ++i)
    check_aggregators(states[i]);
}

TEST_F(ClientJoinTest, JoinAggregatorFailed) {
  std::string aggr_address = "127.0.0.1";
  uint16_t aggr_rpc_port = 8080;
  uint16_t aggr_publish_port = 8081;

  coordinator_.set_fail_requests(true);
  EXPECT_FALSE(states[0].join(this, coordinator_.get_rpc_address(),
                              coordinator_.get_pubsub_address(), &aggr_address,
                              &aggr_rpc_port, &aggr_publish_port));
}
