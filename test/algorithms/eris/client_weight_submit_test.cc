#include "algorithms/eris/client.h"
#include "algorithms/eris/client_test_helpers.h"
#include "algorithms/eris/common.pb.h"
#include <array>
#include <cstddef>
#include <grpcpp/server_context.h>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>

static const size_t aggregator_count = 5;

class ClientWeightSubmitTest : public testing::Test, public ErisMockClient {
protected:
  ClientWeightSubmitTest(void) : ErisMockClient{100} {}

  ~ClientWeightSubmitTest(void) {}

  std::array<MockAggregator, aggregator_count> aggregators_;
  std::array<ClientState, aggregator_count> states_;
};

TEST_F(ClientWeightSubmitTest, SubmitWeights) {
  InitialState state;

  state.mutable_options()->set_rounds(0);
  state.mutable_options()->set_splits(aggregators_.size());
  state.mutable_options()->set_split_seed(42);
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

  for (size_t i = 0; i < states_.size(); ++i)
    EXPECT_TRUE(states_[i].submit_weights(get_parameters(), 0));

  for (size_t i = 0; i < aggregators_.size(); ++i)
    EXPECT_EQ(aggregators_[i].get_received(), states_.size());
}

TEST_F(ClientWeightSubmitTest, OneAggregatorJoinLater) {
  InitialState state;

  state.mutable_options()->set_rounds(0);
  state.mutable_options()->set_splits(aggregators_.size());
  state.mutable_options()->set_split_seed(42);
  state.mutable_options()->set_min_clients(states_.size());

  std::vector<FragmentInfo> infos;
  infos.resize(aggregators_.size());

  for (size_t i = 0; i < aggregators_.size(); ++i) {
    infos[i].set_id(i);
    infos[i].set_submit_address(aggregators_[0].get_rpc_address());
    infos[i].set_publish_address(aggregators_[0].get_pubsub_address());

    if (i > 0)
      *state.add_aggregators() = infos[i];
  }

  for (size_t i = 0; i < states_.size(); ++i)
    states_[i].configure(this, state);

  std::vector<std::thread> threads;
  threads.reserve(states_.size());

  for (size_t i = 0; i < states_.size(); ++i)
    threads.emplace_back(
        [this](ClientState *state) {
          EXPECT_TRUE(state->submit_weights(get_parameters(), 0));
        },
        &states_[i]);

  for (size_t i = 0; i < states_.size(); ++i) {
    std::lock_guard<ClientState> lk(states_[i]);
    states_[i].register_aggregator(infos[0]);
  }

  for (auto &thread : threads)
    thread.join();
}

TEST_F(ClientWeightSubmitTest, SubmitWhileAggregatorsJoining) {
  InitialState state;

  state.mutable_options()->set_rounds(0);
  state.mutable_options()->set_splits(aggregators_.size());
  state.mutable_options()->set_split_seed(42);
  state.mutable_options()->set_min_clients(states_.size());

  std::vector<FragmentInfo> infos;
  infos.resize(aggregators_.size());

  for (size_t i = 0; i < aggregators_.size(); ++i) {
    infos[i].set_id(i);
    infos[i].set_submit_address(aggregators_[0].get_rpc_address());
    infos[i].set_publish_address(aggregators_[0].get_pubsub_address());
  }

  for (size_t i = 0; i < states_.size(); ++i)
    states_[i].configure(this, state);

  std::vector<std::thread> threads;
  threads.reserve(states_.size());

  for (size_t i = 0; i < states_.size(); ++i)
    threads.emplace_back(
        [this](ClientState *state) {
          EXPECT_TRUE(state->submit_weights(get_parameters(), 0));
        },
        &states_[i]);
  for (size_t j = 0; j < infos.size(); ++j)
    for (size_t i = 0; i < states_.size(); ++i) {
      std::lock_guard<ClientState> lk(states_[i]);
      states_[i].register_aggregator(infos[j]);
    }

  for (auto &thread : threads)
    thread.join();
}
