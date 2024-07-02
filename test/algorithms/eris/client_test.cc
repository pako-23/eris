#include "algorithms/eris/aggregator.grpc.pb.h"
#include "algorithms/eris/builder.h"
#include "algorithms/eris/client.h"
#include "algorithms/eris/common.pb.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "algorithms/eris/service.h"
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <grpcpp/server_context.h>
#include <grpcpp/support/server_callback.h>
#include <grpcpp/support/status.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>

using eris::InitialState;
using eris::JoinRequest;
using eris::TrainingOptions;
using grpc::CallbackServerContext;

static constexpr std::chrono::minutes timeout = std::chrono::minutes(1);
static const size_t clients = 5;
static const uint32_t min_clients = 2;
static const uint32_t rounds = 10;
static const uint32_t split_seed = 42;
static const uint32_t splits = 10;

class MockCoordinator {
public:
  MockCoordinator(void)
      : thread_{nullptr}, service_{MockCoordinatorBuilder{}, this},
        fail_request_{false}, aggregators_{} {
    options_.set_min_clients(min_clients);
    options_.set_rounds(rounds);
    options_.set_split_seed(split_seed);
    options_.set_splits(splits);
    aggregators_.resize(options_.splits());

    thread_ = std::make_unique<std::thread>([this]() { service_.start(); });

    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
        get_rpc_address(), grpc::InsecureChannelCredentials());

    bool connected =
        channel->WaitForConnected(std::chrono::system_clock::now() + timeout);

    if (!connected)
      throw std::runtime_error{"failed to setup mock coordinator"};
  }

  ~MockCoordinator(void) {
    service_.stop();
    thread_->join();
  }

  inline std::string get_rpc_address(void) const {
    return "127.0.0.1:" + std::to_string(service_.get_rpc_port());
  }

  inline std::string get_pubsub_address(void) const {
    return "tcp://127.0.0.1:" + std::to_string(service_.get_publish_port());
  }

  void set_fail_requests(bool fail) { fail_request_ = true; }

  void add_aggregator(uint32_t id, const std::string &submit_address,
                      const std::string &publish_address) {
    if (id >= options_.splits())
      return;

    FragmentInfo info;

    info.set_id(id);
    info.set_submit_address(submit_address);
    info.set_publish_address(publish_address);

    aggregators_[id] = std::make_optional<FragmentInfo>(info);
  }

  inline const std::optional<FragmentInfo> &get_aggregator(size_t i) const {
    return aggregators_[i];
  }

  inline const TrainingOptions &get_options(void) const { return options_; }

private:
  class MockCoordinatorBuilder : public ErisServiceBuilder {
  public:
    explicit MockCoordinatorBuilder(void) : ErisServiceBuilder{} {
      add_rpc_port(0);
      add_publish_port(0);
    }
  };

  class MockCoordinatorService : public eris::Coordinator::CallbackService {
  public:
    explicit MockCoordinatorService(MockCoordinator *coordinator)
        : coordinator_{coordinator} {}

    grpc::ServerUnaryReactor *Join(CallbackServerContext *ctx,
                                   const JoinRequest *req,
                                   InitialState *res) override {
      class Reactor : public grpc::ServerUnaryReactor {
      public:
        explicit Reactor(MockCoordinator *coordinator, const JoinRequest *req,
                         InitialState *res) {
          if (coordinator->fail_request_) {
            Finish(grpc::Status(grpc::StatusCode::INTERNAL, "Error"));
            return;
          }

          *res->mutable_options() = coordinator->options_;
          for (const auto &aggr : coordinator->aggregators_)
            if (aggr)
              *res->add_aggregators() = *aggr;

          if (req->has_submit_address()) {
            for (uint32_t i = 0; i < coordinator->aggregators_.size(); ++i)
              if (!coordinator->aggregators_[i]) {
                FragmentInfo info;

                info.set_submit_address(req->submit_address());
                info.set_id(i);
                info.set_publish_address(req->publish_address());

                coordinator->aggregators_[i] =
                    std::make_optional<FragmentInfo>(info);

                res->set_assigned_fragment(i);
                *res->add_aggregators() = info;
                coordinator->service_.publish(info);
                break;
              }
          }

          Finish(grpc::Status::OK);
        }

        void OnDone(void) override { delete this; }
      };

      return new Reactor(coordinator_, req, res);
    };

  private:
    MockCoordinator *coordinator_;
  };

  std::unique_ptr<std::thread> thread_;
  ErisService<MockCoordinatorService> service_;
  TrainingOptions options_;
  bool fail_request_;
  std::vector<std::optional<FragmentInfo>> aggregators_;
};

class ErisMockClient : public ErisClient {
public:
  explicit ErisMockClient(void) : ErisClient{} {}

  std::vector<double> get_parameters(void) {
    std::default_random_engine rng(time(NULL));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> weigths(100);

    for (int i = 0; i < 100; ++i)
      weigths[i] = dist(rng);

    return weigths;
  }
  void set_parameters(const std::vector<double> &parameters) {}
  void fit(void) {}
  void evaluate(void) {}

  bool join(void) { return ErisClient::join(); }

  inline const TrainingOptions &options(void) const { return options_; }

  inline const std::vector<void *> &subscriptions(void) const {
    return subscriptions_;
  }

  inline const std::vector<std::unique_ptr<eris::Aggregator::Stub>> &
  submitters(void) const {
    return submitters_;
  }

  inline bool is_aggregator(void) const {
    return aggregator_ != nullptr && aggregator_thread_ != nullptr;
  }
};

class ErisClientTest : public testing::Test {
protected:
  ErisClientTest(void) : coordinator_{}, rng(time(NULL)), dist(0, splits - 4) {
    for (size_t i = 0; i < clients; ++i) {
      clients_[i].set_coordinator_rpc(coordinator_.get_rpc_address());
      clients_[i].set_coordinator_subscription(
          coordinator_.get_pubsub_address());
    }
  }

  ~ErisClientTest(void) {}

  void check_aggregators(ErisMockClient &client) {
    for (size_t i = 0; i < client.subscriptions().size(); ++i)
      if (!coordinator_.get_aggregator(i)) {
        EXPECT_EQ(client.subscriptions()[i], nullptr);
        EXPECT_EQ(client.submitters()[i], nullptr);
      } else {
        EXPECT_NE(client.subscriptions()[i], nullptr);
        EXPECT_NE(client.submitters()[i], nullptr);
      }
  }

  void check_join(ErisMockClient &client) {
    EXPECT_TRUE(client.join());
    EXPECT_EQ(client.options().min_clients(),
              coordinator_.get_options().min_clients());
    EXPECT_EQ(client.options().rounds(), coordinator_.get_options().rounds());
    EXPECT_EQ(client.options().split_seed(),
              coordinator_.get_options().split_seed());
    EXPECT_EQ(client.options().splits(), coordinator_.get_options().splits());

    EXPECT_EQ(client.submitters().size(), client.options().splits());
    EXPECT_EQ(client.subscriptions().size(), client.options().splits());
    check_aggregators(client);
  }

  MockCoordinator coordinator_;
  ErisMockClient clients_[clients];

  std::default_random_engine rng;
  std::uniform_int_distribution<uint32_t> dist;
};

TEST_F(ErisClientTest, JoinMissingRPCAddress) {
  ErisMockClient client;

  client.set_coordinator_subscription("tcp://127.0.0.0:5000");
  ASSERT_FALSE(client.join());
}

TEST_F(ErisClientTest, JoinMissingPublishAddress) {
  ErisMockClient client;

  client.set_coordinator_rpc("127.0.0.0:5000");
  ASSERT_FALSE(client.join());
}

TEST_F(ErisClientTest, SetInvalidRPCAddress) {
  ErisMockClient client;

  ASSERT_TRUE(client.set_coordinator_subscription("tcp://127.0.0.1:1231"));
  ASSERT_FALSE(client.set_coordinator_rpc("invalid address"));
  ASSERT_FALSE(client.join());
}

TEST_F(ErisClientTest, SetInvalidPublishAddress) {
  ErisMockClient client;

  ASSERT_TRUE(client.set_coordinator_rpc("127.0.0.1:1231"));
  ASSERT_FALSE(client.set_coordinator_subscription("invalid address"));
  ASSERT_FALSE(client.join());
}

TEST_F(ErisClientTest, SetInvalidAggregatorConfig) {
  ASSERT_FALSE(clients_[0].set_aggregator_config("127.0.1", 1231, 1323));
  ASSERT_FALSE(clients_[0].set_aggregator_config("0.0.0.0", 1231, 1323));
  ASSERT_FALSE(clients_[0].set_aggregator_config("127.0.0.1", 1231, 0));
  ASSERT_FALSE(clients_[0].set_aggregator_config("127.0.0.1", 0, 1323));
}

TEST_F(ErisClientTest, JoinClient) {
  uint32_t size = dist(rng);

  for (uint32_t i = 0; i < size; ++i)
    coordinator_.add_aggregator(i, "127.0.0.1:50051", "tcp://127.0.0.1:5555");

  for (size_t i = 0; i < clients; ++i)
    check_join(clients_[i]);
}

TEST_F(ErisClientTest, JoinAggregator) {
  uint32_t size = dist(rng);

  for (uint32_t i = 0; i < size; ++i)
    coordinator_.add_aggregator(i, "127.0.0.1:50051", "tcp://127.0.0.1:5555");

  for (size_t i = 1; i < clients; ++i)
    check_join(clients_[i]);

  EXPECT_TRUE(clients_[0].set_aggregator_config("127.0.0.1", 8080, 8081));
  check_join(clients_[0]);
  ASSERT_TRUE(clients_[0].is_aggregator());

  for (size_t i = 1; i < clients; ++i)
    check_aggregators(clients_[i]);
}

TEST_F(ErisClientTest, JoinAggregatorNoFragmentAssigned) {
  for (uint32_t i = 0; i < coordinator_.get_options().splits(); ++i)
    coordinator_.add_aggregator(i, "127.0.0.1:50051", "tcp://127.0.0.1:5555");

  for (size_t i = 1; i < clients; ++i)
    check_join(clients_[i]);

  EXPECT_TRUE(clients_[0].set_aggregator_config("127.0.0.1", 8080, 8081));
  check_join(clients_[0]);
  ASSERT_FALSE(clients_[0].is_aggregator());

  for (size_t i = 1; i < clients; ++i)
    check_aggregators(clients_[i]);
}

TEST_F(ErisClientTest, JoinClientFailed) {
  coordinator_.set_fail_requests(true);
  EXPECT_FALSE(clients_[0].join());
}

TEST_F(ErisClientTest, JoinAggregatorFailed) {
  coordinator_.set_fail_requests(true);
  EXPECT_TRUE(clients_[0].set_aggregator_config("127.0.0.1", 8080, 8081));
  EXPECT_FALSE(clients_[0].join());
}
