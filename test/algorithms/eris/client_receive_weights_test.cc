#include "algorithms/eris/aggregator.pb.h"
#include "algorithms/eris/client_test_helpers.h"
#include "algorithms/eris/split.h"
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <grpcpp/server_context.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <thread>
#include <vector>

static const uint32_t aggregator_count = 5;
static const uint32_t split_seed = 42;
static const size_t parameters_size = 130;

class ClientReceiveWeightsTest : public testing::Test, public MockClient {
protected:
  ClientReceiveWeightsTest(void)
      : MockClient{parameters_size}, expected_{generate_random_vector()} {
    splitter.configure(expected_, aggregator_count, split_seed);

    InitialState state;

    state.mutable_options()->set_rounds(10);
    state.mutable_options()->set_splits(aggregators_.size());
    state.mutable_options()->set_split_seed(split_seed);
    state.mutable_options()->set_min_clients(states_.size());

    for (size_t i = 0; i < aggregators_.size(); ++i) {
      FragmentInfo info;

      info.set_id(i);
      info.set_submit_address(aggregators_[i].get_rpc_address());
      info.set_publish_address(aggregators_[i].get_pubsub_address());

      *state.add_aggregators() = info;
    }

    for (size_t i = 0; i < states_.size(); ++i)
      states_[i].configure(this, state);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  std::vector<double> generate_random_vector(void) {
    std::vector<double> vec(parameters_size);
    std::default_random_engine rng(time(NULL));
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < vec.size(); ++i)
      vec[i] = dist(rng);

    return vec;
  }

  std::vector<eris::WeightUpdate> generate_updates(uint32_t round) {
    std::vector<eris::FragmentWeights> updates =
        splitter.split(expected_, round);
    std::vector<eris::WeightUpdate> result(updates.size());

    for (size_t i = 0; i < updates.size(); ++i) {

      result[i].set_round(updates[i].round());
      result[i].set_contributors(1);

      for (int j = 0; j < updates[i].weight_size(); ++j)
        result[i].add_weight(updates[i].weight(j));
    }

    return result;
  }

  ~ClientReceiveWeightsTest(void) {}

  std::array<MockAggregator, aggregator_count> aggregators_;
  std::array<ClientState, 8> states_;
  std::vector<double> expected_;
  RandomSplit splitter;
};

TEST_F(ClientReceiveWeightsTest, ReceiveWeights) {
  std::vector<std::thread> threads;
  threads.reserve(states_.size());

  for (size_t i = 0; i < states_.size(); ++i)
    threads.emplace_back(
        [this](ClientState *state) {
          uint32_t round = 0;
          std::vector<double> updates = state->receive_weights(&round);

          EXPECT_EQ(updates.size(), expected_.size());

          for (size_t i = 0; i < updates.size(); ++i)
            EXPECT_NEAR(updates[i], expected_[i],
                        5 * std::numeric_limits<double>::epsilon())
                << "Elements are different at index " << i;
          EXPECT_EQ(round, 0);
        },
        &states_[i]);

  std::vector<WeightUpdate> updates = generate_updates(0);

  for (size_t i = 0; i < updates.size(); ++i)
    aggregators_[i].publish_update(updates[i]);

  for (auto &thread : threads)
    thread.join();
}

TEST_F(ClientReceiveWeightsTest, ReceiveOlderWeights) {
  std::vector<std::thread> threads;
  threads.reserve(states_.size());

  for (size_t i = 0; i < states_.size(); ++i)
    threads.emplace_back(
        [this](ClientState *state) {
          uint32_t round = 1;
          std::vector<double> updates = state->receive_weights(&round);

          EXPECT_EQ(updates.size(), expected_.size());

          for (size_t i = 0; i < updates.size(); ++i)
            EXPECT_NEAR(updates[i], expected_[i],
                        5 * std::numeric_limits<double>::epsilon())
                << "Elements are different at index " << i;
          EXPECT_EQ(round, 1);
        },
        &states_[i]);

  std::vector<WeightUpdate> updates = generate_updates(0);

  for (size_t i = 0; i < updates.size(); ++i)
    aggregators_[i].publish_update(updates[i]);

  generate_random_vector();
  updates = generate_updates(1);

  for (size_t i = 0; i < updates.size(); ++i)
    aggregators_[i].publish_update(updates[i]);

  for (auto &thread : threads)
    thread.join();
}

TEST_F(ClientReceiveWeightsTest, ReceiveNewerWeights) {
  std::vector<std::thread> threads;
  threads.reserve(states_.size());

  for (size_t i = 0; i < states_.size(); ++i)
    threads.emplace_back(
        [this](ClientState *state) {
          uint32_t round = 0;
          std::vector<double> updates = state->receive_weights(&round);

          EXPECT_EQ(updates.size(), expected_.size());

          for (size_t i = 0; i < updates.size(); ++i)
            EXPECT_NEAR(updates[i], expected_[i],
                        5 * std::numeric_limits<double>::epsilon())
                << "Elements are different at index " << i;

          EXPECT_EQ(round, 1);
        },
        &states_[i]);

  std::vector<WeightUpdate> updates = generate_updates(0);

  std::default_random_engine rng(time(NULL));
  std::uniform_int_distribution<size_t> dist(0, updates.size() - 1);

  size_t excluded = dist(rng);

  for (size_t i = 0; i < updates.size(); ++i)
    if (i == excluded)
      aggregators_[i].publish_update(updates[i]);

  generate_random_vector();
  updates = generate_updates(1);

  for (size_t i = 0; i < updates.size(); ++i)
    aggregators_[i].publish_update(updates[i]);

  for (auto &thread : threads)
    thread.join();
}

TEST_F(ClientReceiveWeightsTest, ReceiveNewerWeightsMultipleTimes) {
  std::vector<std::thread> threads;
  threads.reserve(states_.size());

  for (size_t i = 0; i < states_.size(); ++i)
    threads.emplace_back(
        [this](ClientState *state) {
          uint32_t round = 0;
          std::vector<double> updates = state->receive_weights(&round);

          EXPECT_EQ(updates.size(), expected_.size());

          for (size_t i = 0; i < updates.size(); ++i)
            EXPECT_NEAR(updates[i], expected_[i],
                        5 * std::numeric_limits<double>::epsilon())
                << "Elements are different at index " << i;

          EXPECT_EQ(round, 2);
        },
        &states_[i]);

  std::vector<WeightUpdate> updates = generate_updates(0);

  std::default_random_engine rng(time(NULL));
  std::uniform_int_distribution<size_t> dist(0, updates.size() - 1);

  size_t excluded = dist(rng);

  for (size_t i = 0; i < updates.size(); ++i)
    if (i == excluded)
      aggregators_[i].publish_update(updates[i]);

  generate_random_vector();
  updates = generate_updates(1);
  excluded = dist(rng);

  for (size_t i = 0; i < updates.size(); ++i)
    if (i == excluded)
      aggregators_[i].publish_update(updates[i]);

  generate_random_vector();
  updates = generate_updates(2);

  for (size_t i = 0; i < updates.size(); ++i)
    aggregators_[i].publish_update(updates[i]);

  for (auto &thread : threads)
    thread.join();
}
