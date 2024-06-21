#include "algorithms/eris/builder.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <grpc/grpc.h>
#include <grpc/support/port_platform.h>
#include <grpc/support/time.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/support/client_callback.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/stub_options.h>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>

using eris::CoordinatorUpdate;

static constexpr std::chrono::minutes timeout = std::chrono::minutes(1);
static const uint32_t min_clients = 2;
static const uint32_t rounds = 10;
static const uint32_t split_seed = 42;
static const uint32_t splits = 5;

static void validate_training_options(const CoordinatorUpdate &res) {
  ASSERT_TRUE(res.has_init_config());

  ASSERT_EQ(res.init_config().options().min_clients(), min_clients);
  ASSERT_EQ(res.init_config().options().rounds(), rounds);
  ASSERT_EQ(res.init_config().options().split_seed(), split_seed);
  ASSERT_EQ(res.init_config().options().splits(), splits);
}

class ErisCoordinatorTest : public testing::Test {
protected:
  ErisCoordinatorTest(void) : server_{nullptr}, server_thread_{nullptr} {
    ErisCoordinatorBuilder builder;
    builder.add_min_clients(min_clients);
    builder.add_rounds(rounds);
    builder.add_rpc_port(0);
    builder.add_split_seed(split_seed);
    builder.add_splits(splits);

    server_ = std::make_shared<ErisCoordinator>(builder);
    server_port_ = server_->get_listening_port();
    server_thread_ = std::make_unique<std::thread>(
        [](std::shared_ptr<ErisCoordinator> coordinator) {
          coordinator->start();
        },
        server_);

    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
        get_server_address(), grpc::InsecureChannelCredentials());

    bool connected =
        channel->WaitForConnected(std::chrono::system_clock::now() + timeout);

    if (!connected) {
      server_->stop();
      server_thread_->join();
      server_ = nullptr;
      server_thread_ = nullptr;
    }
  }

  ~ErisCoordinatorTest(void) {
    if (server_)
      server_->stop();
    if (server_thread_)
      server_thread_->join();
  }

  std::string get_server_address(void) const {
    return "127.0.0.1:" + std::to_string(server_port_);
  }

  std::unique_ptr<eris::Coordinator::Stub> connect(void) {
    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
        get_server_address(), grpc::InsecureChannelCredentials());
    return eris::Coordinator::NewStub(channel);
  }

  std::shared_ptr<ErisCoordinator> server_;
  std::unique_ptr<std::thread> server_thread_;
  uint16_t server_port_;
};

class TestClient : public grpc::ClientReadReactor<CoordinatorUpdate> {
public:
  TestClient(eris::Coordinator::Stub *stub, const eris::JoinRequest &req,
             std::function<void(bool, const CoordinatorUpdate &)> check)
      : check_{check} {
    stub->async()->Join(&ctx_, &req, this);
    StartRead(&res_);
    StartCall();
  }

  void OnReadDone(bool ok) override { check_(ok, res_); }

  void OnDone(const Status &s) override {
    std::lock_guard<std::mutex> lk(mu_);
    status_ = s;
    done_ = true;
    cv_.notify_one();
  }

  Status Await() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] { return done_; });
    return std::move(status_);
  }

private:
  CoordinatorUpdate res_;
  grpc::ClientContext ctx_;

  std::function<void(bool, const CoordinatorUpdate &)> check_;
  std::mutex mu_;
  std::condition_variable cv_;
  Status status_;
  bool done_ = false;
};

TEST_F(ErisCoordinatorTest, Initialization) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);
}

TEST_F(ErisCoordinatorTest, JoinClient) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);

  eris::JoinRequest req;

  std::unique_ptr<eris::Coordinator::Stub> stub = connect();

  TestClient cli(stub.get(), req, [](bool ok, const CoordinatorUpdate &res) {
    ASSERT_TRUE(ok);
    ASSERT_FALSE(res.has_aggregator());
    validate_training_options(res);
    ASSERT_EQ(res.init_config().aggregators_size(), 0);
  });
  grpc::Status status = cli.Await();
  ASSERT_TRUE(status.ok());
}

TEST_F(ErisCoordinatorTest, JoinAggregator) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);

  const std::string aggregation_address = "127.0.0.0:50052";

  eris::JoinRequest req;
  *req.mutable_aggr_address() = aggregation_address;

  std::unique_ptr<eris::Coordinator::Stub> stub = connect();

  TestClient cli(stub.get(), req,
                 [&aggregation_address](bool ok, const CoordinatorUpdate &res) {
                   ASSERT_TRUE(ok);
                   ASSERT_FALSE(res.has_aggregator());

                   validate_training_options(res);

                   ASSERT_EQ(res.init_config().aggregators_size(), 1);
                   ASSERT_TRUE(res.init_config().has_assigned_fragment());
                   ASSERT_GE(res.init_config().assigned_fragment(), 0);
                   ASSERT_LT(res.init_config().assigned_fragment(), splits);

                   bool found = false;

                   for (auto aggr : res.init_config().aggregators())
                     if (aggr.aggregator() == aggregation_address)
                       found = true;

                   ASSERT_TRUE(found);
                 });
  grpc::Status status = cli.Await();
  ASSERT_TRUE(status.ok());
}

TEST_F(ErisCoordinatorTest, JoinTooManyAggregators) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);

  std::unordered_set<std::string> aggregators;

  const uint16_t base_port = 50052;
  for (uint16_t i = 0; i < splits; ++i)
    aggregators.insert("127.0.0.0:" + std::to_string(base_port + i));

  std::vector<std::thread> threads;
  threads.reserve(aggregators.size());

  for (const std::string &address : aggregators)
    threads.emplace_back(
        [this](const std::string &address) {
          eris::JoinRequest req;
          *req.mutable_aggr_address() = address;

          std::unique_ptr<eris::Coordinator::Stub> stub = connect();

          TestClient cli(
              stub.get(), req,
              [address](bool ok, const CoordinatorUpdate &res) {
                ASSERT_TRUE(ok);
                ASSERT_FALSE(res.has_aggregator());

                validate_training_options(res);

                ASSERT_GT(res.init_config().aggregators_size(), 0);

                ASSERT_TRUE(res.init_config().has_assigned_fragment());
                ASSERT_GE(res.init_config().assigned_fragment(), 0);
                ASSERT_LT(res.init_config().assigned_fragment(), splits);

                bool found = false;

                for (auto aggr : res.init_config().aggregators())
                  if (aggr.aggregator() == address)
                    found = true;

                ASSERT_TRUE(found);
              });
          grpc::Status status = cli.Await();
          ASSERT_TRUE(status.ok());
        },
        address);

  for (auto &t : threads)
    t.join();

  eris::JoinRequest req;
  *req.mutable_aggr_address() =
      "127.0.0.0:" + std::to_string(base_port + aggregators.size() + 1);

  std::unique_ptr<eris::Coordinator::Stub> stub = connect();

  TestClient cli(
      stub.get(), req, [&aggregators](bool ok, const CoordinatorUpdate &res) {
        ASSERT_TRUE(ok);
        ASSERT_FALSE(res.has_aggregator());

        validate_training_options(res);

        ASSERT_EQ(res.init_config().aggregators_size(), splits);

        ASSERT_FALSE(res.init_config().has_assigned_fragment());
        ASSERT_EQ(aggregators.size(), res.init_config().aggregators_size());

        for (auto aggr : res.init_config().aggregators())
          if (aggregators.find(aggr.aggregator()) == aggregators.end())
            FAIL();
      });
  grpc::Status status = cli.Await();
  ASSERT_TRUE(status.ok());
}

TEST_F(ErisCoordinatorTest, InvalidAggregatorAddress) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);

  eris::JoinRequest req;
  *req.mutable_aggr_address() = "Some random string";

  std::unique_ptr<eris::Coordinator::Stub> stub = connect();

  TestClient cli(stub.get(), req, [](bool ok, const CoordinatorUpdate &res) {
    ASSERT_FALSE(ok);
  });
  grpc::Status status = cli.Await();
  ASSERT_FALSE(status.ok());

  ASSERT_EQ(status.error_code(), StatusCode::INVALID_ARGUMENT);
  ASSERT_STREQ(status.error_message().c_str(),
               "An aggregator address must have the form <address>:<port> "
               "where address is a valid IPv4 address");
}

// TEST_F(ErisCoordinatorTest, ReceiveUpdates) {
//   ASSERT_NE(server_, nullptr);
//   ASSERT_NE(server_thread_, nullptr);

//   const uint16_t base_port = 50052;
//   std::unordered_set<std::string> aggregators;

//   for (uint16_t i = 0; i < splits; ++i)
//     aggregators.insert("127.0.0.0:" + std::to_string(base_port + i));

//   std::vector<std::thread> threads;
//   threads.reserve(aggregators.size());

//   for (const std::string &address : aggregators)
//     threads.emplace_back(
//         [this](const std::string &address) {
//           eris::JoinRequest req;
//           *req.mutable_aggr_address() = address;

//           std::unique_ptr<eris::Coordinator::Stub> stub = connect();

//           TestClient cli(stub.get(), req,
//                          [address](bool ok, const CoordinatorUpdate &res)
//                          {});
//           grpc::Status status = cli.Await();
//           ASSERT_TRUE(status.ok());
//         },
//         address);

//   for (auto &t : threads)
//     t.join();

//   eris::JoinRequest req;
//   *req.mutable_aggr_address() =
//       "127.0.0.0:" + std::to_string(base_port + aggregators.size() + 1);

//   std::unique_ptr<eris::Coordinator::Stub> stub = connect();

//   TestClient cli(
//       stub.get(), req, [&aggregators](bool ok, const CoordinatorUpdate &res)
//       {
//         ASSERT_TRUE(ok);
//         ASSERT_FALSE(res.has_aggregator());

//         validate_training_options(res);

//         ASSERT_EQ(res.init_config().aggregators_size(), splits);

//         ASSERT_FALSE(res.init_config().has_assigned_fragment());
//         ASSERT_EQ(aggregators.size(), res.init_config().aggregators_size());

//         for (auto aggr : res.init_config().aggregators())
//           if (aggregators.find(aggr.aggregator()) == aggregators.end())
//             FAIL();
//       });
//   grpc::Status status = cli.Await();
//   ASSERT_TRUE(status.ok());
// }
