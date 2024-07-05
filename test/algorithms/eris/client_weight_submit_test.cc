#include "algorithms/eris/aggregator.grpc.pb.h"
#include "algorithms/eris/aggregator.pb.h"
#include "algorithms/eris/builder.h"
#include "algorithms/eris/client.h"
#include "algorithms/eris/client_test_helpers.h"
#include "algorithms/eris/common.pb.h"
#include "algorithms/eris/service.h"
#include <array>
#include <atomic>
#include <cstddef>
#include <grpcpp/server_context.h>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>

using eris::FragmentWeights;
using grpc::CallbackServerContext;

static const size_t aggregator_count = 5;

class MockAggregator final {
public:
  explicit MockAggregator(void)
      : service_{MockAggregatorBuilder{}, this}, received{0} {

    thread_ = std::make_unique<std::thread>([this]() { service_.start(); });

    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
        get_rpc_address(), grpc::InsecureChannelCredentials());

    bool connected = channel->WaitForConnected(
        std::chrono::system_clock::now() + std::chrono::minutes(1));

    if (!connected)
      throw std::runtime_error{"failed to setup mock coordinator"};
  }

  ~MockAggregator(void) {
    service_.stop();
    thread_->join();
  }

  inline std::string get_rpc_address(void) const {
    return "127.0.0.1:" + std::to_string(service_.get_rpc_port());
  }

  inline std::string get_pubsub_address(void) const {
    return "tcp://127.0.0.1:" + std::to_string(service_.get_publish_port());
  }

  inline size_t get_received(void) const { return received; }

private:
  class MockAggregatorBuilder : public ErisServiceBuilder {
  public:
    explicit MockAggregatorBuilder(void) : ErisServiceBuilder{} {
      add_rpc_port(0);
      add_publish_port(0);
    }
  };

  class MockAggregatorService : public eris::Aggregator::CallbackService {
  public:
    explicit MockAggregatorService(MockAggregator *aggr) noexcept
        : aggr_{aggr} {}

    grpc::ServerUnaryReactor *SubmitWeights(CallbackServerContext *ctx,
                                            const FragmentWeights *req,
                                            eris::Empty *res) override {
      class Reactor : public grpc::ServerUnaryReactor {
      public:
        explicit Reactor(MockAggregator *aggr) {
          ++aggr->received;
          Finish(grpc::Status::OK);
        }

        void OnDone(void) override { delete this; }
      };

      return new Reactor(aggr_);
    }

  private:
    MockAggregator *aggr_;
  };

  ErisService<MockAggregatorService> service_;
  std::unique_ptr<std::thread> thread_;

  std::atomic_size_t received;
};

class ClientWeightSubmitTest : public testing::Test, public ErisMockClient {
protected:
  ClientWeightSubmitTest(void) {}

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
