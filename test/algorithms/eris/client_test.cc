#include "algorithms/eris/client_test_helpers.h"
#include "algorithms/eris/split.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <utility>
#include <vector>

static const size_t client_count = 10;
static const uint32_t split_seed = 42;
static const uint32_t splits = 5;
static const uint32_t rounds = 3;
static const size_t model_size = 100;

class ErisClientConfigTest : public testing::Test {
protected:
  ErisClientConfigTest(void) : client{100} {}

  MockClient client;
};

class ErisClientTest : public testing::Test {

  template <std::size_t... indexes>
  ErisClientTest(size_t parameters_size,
                 std::index_sequence<indexes...> const &)
      : expected_parameters{},
        clients{((void)indexes,
                 MockClient{parameters_size, &expected_parameters})...},
        splitter(), coordinator{client_count, rounds, split_seed, splits} {}

protected:
  ErisClientTest(void)
      : ErisClientTest{model_size, std::make_index_sequence<client_count>{}} {}
  ~ErisClientTest(void) {}

  void generate_parameters(void) {
    std::default_random_engine rng(time(NULL));
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    expected_parameters.resize(rounds);

    for (size_t i = 0; i < expected_parameters.size(); ++i) {
      expected_parameters[i].resize(model_size);
      for (size_t j = 0; j < expected_parameters[i].size(); ++j)
        expected_parameters[i][j] = dist(rng);
    }

    splitter.configure(expected_parameters[0], splits, split_seed);
    publish_weights.resize(splits);

    for (size_t i = 0; i < expected_parameters.size(); ++i) {
      std::vector<FragmentWeights> fragments =
          splitter.split(expected_parameters[i], i);

      for (size_t j = 0; j < fragments.size(); ++j) {
        WeightUpdate update;

        update.set_round(i);
        update.set_contributors(1);

        for (int k = 0; k < fragments[j].weight_size(); ++k)
          update.add_weight(fragments[j].weight(k));

        publish_weights[j].push_back(update);
      }
    }
  }

  std::vector<std::vector<float>> expected_parameters;
  std::vector<std::vector<WeightUpdate>> publish_weights;
  std::array<MockClient, client_count> clients;
  std::array<MockAggregator, splits> aggregators;
  RandomSplit splitter;
  MockCoordinator coordinator;
};

TEST_F(ErisClientConfigTest, JoinMissingRPCAddress) {
  EXPECT_TRUE(client.set_coordinator_subscription("tcp://127.0.0.0:5000"));
  EXPECT_FALSE(client.train());
}

TEST_F(ErisClientConfigTest, JoinMissingPublishAddress) {
  EXPECT_TRUE(client.set_coordinator_rpc("127.0.0.0:5000"));
  EXPECT_FALSE(client.train());
}

TEST_F(ErisClientConfigTest, SetInvalidRPCAddress) {
  EXPECT_TRUE(client.set_coordinator_subscription("tcp://127.0.0.1:1231"));
  EXPECT_FALSE(client.set_coordinator_rpc("invalid address"));
  EXPECT_FALSE(client.train());
}

TEST_F(ErisClientConfigTest, SetInvalidPublishAddress) {
  EXPECT_TRUE(client.set_coordinator_rpc("127.0.0.1:1231"));
  EXPECT_FALSE(client.set_coordinator_subscription("invalid address"));
  EXPECT_FALSE(client.train());
}

TEST_F(ErisClientConfigTest, SetValidAggregatorConfig) {
  EXPECT_TRUE(client.set_aggregator_config("127.0.0.1", 1231, 120));
}

TEST_F(ErisClientConfigTest, SetInvalidAggregatorConfig) {
  EXPECT_FALSE(client.set_aggregator_config("127.0.1", 1231, 1323));
  EXPECT_FALSE(client.set_aggregator_config("0.0.0.0", 1231, 1323));
  EXPECT_FALSE(client.set_aggregator_config("127.0.0.1", 1231, 0));
  EXPECT_FALSE(client.set_aggregator_config("127.0.0.1", 0, 1323));
  EXPECT_FALSE(client.set_aggregator_config("127.0.0.1", 1231, 1231));
}

TEST_F(ErisClientTest, FailClientJoin) {
  coordinator.set_fail_requests(true);
  EXPECT_TRUE(clients[0].set_coordinator_rpc(coordinator.get_rpc_address()));
  EXPECT_TRUE(clients[0].set_coordinator_subscription(
      coordinator.get_pubsub_address()));
  EXPECT_FALSE(clients[0].train());
}

TEST_F(ErisClientTest, FailAggregatorJoin) {
  coordinator.set_fail_requests(true);
  EXPECT_TRUE(clients[0].set_coordinator_rpc(coordinator.get_rpc_address()));
  EXPECT_TRUE(clients[0].set_coordinator_subscription(
      coordinator.get_pubsub_address()));
  EXPECT_TRUE(clients[0].set_aggregator_config("127.0.0.1", 1231, 120));
  EXPECT_FALSE(clients[0].train());
}

TEST_F(ErisClientTest, Training) {
  generate_parameters();

  for (size_t i = 0; i < clients.size(); ++i) {
    EXPECT_TRUE(clients[i].set_coordinator_rpc(coordinator.get_rpc_address()));
    EXPECT_TRUE(clients[i].set_coordinator_subscription(
        coordinator.get_pubsub_address()));
  }

  std::vector<std::thread> threads;
  threads.reserve(clients.size());

  for (size_t i = 0; i < clients.size(); ++i)
    threads.emplace_back(
        [](MockClient *client) {
          EXPECT_TRUE(client->train());
          EXPECT_EQ(client->get_fit_calls(), rounds);
          EXPECT_EQ(client->get_set_parameters_calls(), rounds);
          EXPECT_EQ(client->get_evaluate_calls(), rounds);
        },
        &clients[i]);

  for (size_t i = 0; i < aggregators.size(); ++i) {
    aggregators[i].set_min_clients(coordinator.get_options().min_clients());
    aggregators[i].set_publish_weights(&publish_weights[i]);
    coordinator.add_aggregator(i, aggregators[i].get_rpc_address(),
                               aggregators[i].get_pubsub_address());
  }

  for (auto &thread : threads)
    thread.join();
}
