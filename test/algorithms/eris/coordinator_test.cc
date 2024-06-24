#include "algorithms/eris/builder.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "zmq.h"
#include <chrono>
#include <condition_variable>
#include <cstddef>
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
#include <vector>

using eris::InitialState;
using eris::JoinRequest;

static constexpr std::chrono::minutes timeout = std::chrono::minutes(1);
static const int zmq_timeout = 500;
static const uint32_t min_clients = 2;
static const uint32_t rounds = 10;
static const uint32_t split_seed = 42;
static const uint32_t splits = 10;

static void validate_training_options(const InitialState &state) {
  ASSERT_EQ(state.options().min_clients(), min_clients);
  ASSERT_EQ(state.options().rounds(), rounds);
  ASSERT_EQ(state.options().split_seed(), split_seed);
  ASSERT_EQ(state.options().splits(), splits);
}

static void test_join(const std::string &coordinator, const JoinRequest &req,
                      InitialState *res,
                      std::function<void(Status, InitialState *)> check) {
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(coordinator, grpc::InsecureChannelCredentials());
  std::unique_ptr<eris::Coordinator::Stub> stub =
      eris::Coordinator::NewStub(channel);

  grpc::ClientContext ctx;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;

  stub->async()->Join(&ctx, &req, res,
                      [&check, &mu, &done, &cv, res](Status s) {
                        check(s, res);
                        std::lock_guard<std::mutex> lk(mu);

                        done = true;
                        cv.notify_one();
                      });

  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [&done] { return done; });
}

class ErisCoordinatorTest : public testing::Test {
protected:
  ErisCoordinatorTest(void) : server_{nullptr}, server_thread_{nullptr} {
    ErisCoordinatorBuilder builder;
    builder.add_min_clients(min_clients);
    builder.add_rounds(rounds);
    builder.add_rpc_port(0);
    builder.add_pubsub_port(0);
    builder.add_split_seed(split_seed);
    builder.add_splits(splits);

    server_ = std::make_shared<ErisCoordinator>(builder);
    server_thread_ = std::make_unique<std::thread>(
        [](std::shared_ptr<ErisCoordinator> coordinator) {
          coordinator->start();
        },
        server_);

    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
        get_rpc_address(), grpc::InsecureChannelCredentials());

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

  void SetUp(void) override {
    for (size_t i = 0; i < subscribers; ++i) {
      ctx[i] = zmq_ctx_new();
      ASSERT_NE(ctx[i], nullptr);
      subscriber[i] = zmq_socket(ctx[i], ZMQ_SUB);
      ASSERT_NE(subscriber[i], nullptr);
      ASSERT_EQ(zmq_connect(subscriber[i], get_pubsub_address().c_str()), 0);
      ASSERT_EQ(zmq_setsockopt(subscriber[i], ZMQ_SUBSCRIBE, "", 0), 0);
      ASSERT_EQ(zmq_setsockopt(subscriber[i], ZMQ_RCVTIMEO, &zmq_timeout,
                               sizeof(zmq_timeout)),
                0);
    }
  }

  void TearDown(void) override {
    for (size_t i = 0; i < subscribers; ++i) {
      zmq_close(subscriber[i]);
      zmq_ctx_destroy(ctx[i]);
    }
  }

  inline std::string get_rpc_address(void) const {
    return "127.0.0.1:" + std::to_string(server_->get_rpc_port());
  }

  inline std::string get_pubsub_address(void) const {
    return "tcp://127.0.0.1:" + std::to_string(server_->get_pubssub_port());
  }

  static const size_t subscribers = 5;
  std::shared_ptr<ErisCoordinator> server_;
  std::unique_ptr<std::thread> server_thread_;
  void *ctx[subscribers];
  void *subscriber[subscribers];
};

TEST_F(ErisCoordinatorTest, Initialization) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);
}

TEST_F(ErisCoordinatorTest, JoinClient) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);

  JoinRequest req;
  InitialState res;

  test_join(get_rpc_address(), req, &res,
            [](Status status, InitialState *state) {
              ASSERT_TRUE(status.ok());
              validate_training_options(*state);
              ASSERT_FALSE(state->has_assigned_fragment());
              ASSERT_EQ(state->aggregators_size(), 0);
            });

  std::vector<std::thread> threads;
  threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    threads.emplace_back(
        [](void *sub) {
          zmq_msg_t msg;
          zmq_msg_init(&msg);
          ASSERT_EQ(zmq_msg_recv(&msg, sub, 0), -1);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  for (auto &thread : threads)
    thread.join();
}

TEST_F(ErisCoordinatorTest, JoinAggregator) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);

  const std::string aggregation_address = "127.0.0.0:50052";

  InitialState res;
  JoinRequest req;
  *req.mutable_aggr_address() = aggregation_address;

  std::vector<std::thread> threads;
  threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    threads.emplace_back(
        [&aggregation_address](void *sub) {
          zmq_msg_t msg;
          FragmentInfo info;

          zmq_msg_init(&msg);
          int size = zmq_msg_recv(&msg, sub, 0);
          ASSERT_NE(size, -1);
          ASSERT_TRUE(info.ParseFromArray(zmq_msg_data(&msg), size));
          ASSERT_EQ(info.aggregator(), aggregation_address);
          ASSERT_LT(info.id(), splits);

          zmq_msg_close(&msg);
        },
        subscriber[i]);

  test_join(get_rpc_address(), req, &res,
            [&aggregation_address](Status status, InitialState *state) {
              ASSERT_TRUE(status.ok());
              validate_training_options(*state);
              ASSERT_TRUE(state->has_assigned_fragment());
              ASSERT_EQ(state->aggregators_size(), 1);
              ASSERT_GE(state->assigned_fragment(), 0);
              ASSERT_LT(state->assigned_fragment(), splits);

              bool found = false;

              for (auto aggr : state->aggregators())
                if (aggr.aggregator() == aggregation_address)
                  found = true;

              ASSERT_TRUE(found);
            });

  for (auto &thread : threads)
    thread.join();
}

TEST_F(ErisCoordinatorTest, JoinTooManyAggregators) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);

  std::unordered_set<std::string> aggregators;

  const uint16_t base_port = 50052;
  for (uint16_t i = 0; i < splits; ++i)
    aggregators.insert("127.0.0.0:" + std::to_string(base_port + i));

  std::vector<std::thread> subscriber_threads;
  subscriber_threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    subscriber_threads.emplace_back(
        [&aggregators](void *sub) {
          zmq_msg_t msg;
          FragmentInfo info;

          std::unordered_set<std::string> addresses;
          std::unordered_set<uint32_t> ids;

          for (size_t i = 0; i < aggregators.size(); ++i) {
            zmq_msg_init(&msg);
            int size = zmq_msg_recv(&msg, sub, 0);
            ASSERT_NE(size, -1);
            ASSERT_TRUE(info.ParseFromArray(zmq_msg_data(&msg), size));
            ASSERT_NE(aggregators.find(info.aggregator()), aggregators.end());
            ASSERT_LT(info.id(), splits);
            addresses.insert(info.aggregator());
            ids.insert(info.id());
            zmq_msg_close(&msg);
          }

          ASSERT_EQ(addresses.size(), aggregators.size());
          ASSERT_EQ(ids.size(), splits);

          zmq_msg_init(&msg);
          ASSERT_EQ(zmq_msg_recv(&msg, sub, 0), -1);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  std::vector<std::thread> threads;
  threads.reserve(aggregators.size());

  for (const std::string &address : aggregators)
    threads.emplace_back(
        [this](const std::string &address) {
          InitialState res;
          JoinRequest req;
          *req.mutable_aggr_address() = address;

          test_join(get_rpc_address(), req, &res,
                    [&address](Status status, InitialState *state) {
                      ASSERT_TRUE(status.ok());
                      validate_training_options(*state);
                      ASSERT_TRUE(state->has_assigned_fragment());
                      ASSERT_GE(state->assigned_fragment(), 0);
                      ASSERT_LT(state->assigned_fragment(), splits);
                      bool found = false;

                      for (auto aggr : state->aggregators())
                        if (aggr.aggregator() == address)
                          found = true;

                      ASSERT_TRUE(found);
                    });
        },
        address);

  for (auto &thread : threads)
    thread.join();

  InitialState res;
  JoinRequest req;
  *req.mutable_aggr_address() =
      "127.0.0.0:" + std::to_string(base_port + aggregators.size() + 1);

  test_join(get_rpc_address(), req, &res,
            [&aggregators](Status status, InitialState *state) {
              ASSERT_TRUE(status.ok());
              validate_training_options(*state);
              ASSERT_FALSE(state->has_assigned_fragment());

              ASSERT_EQ(aggregators.size(), state->aggregators_size());

              for (auto aggr : state->aggregators())
                if (aggregators.find(aggr.aggregator()) == aggregators.end())
                  FAIL();
            });

  for (auto &thread : subscriber_threads)
    thread.join();
}

TEST_F(ErisCoordinatorTest, InvalidAggregatorAddress) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);

  JoinRequest req;
  InitialState res;
  *req.mutable_aggr_address() = "Some random string";

  test_join(get_rpc_address(), req, &res,
            [](Status status, InitialState *state) {
              ASSERT_FALSE(status.ok());
              ASSERT_EQ(status.error_code(), StatusCode::INVALID_ARGUMENT);
              ASSERT_STREQ(
                  status.error_message().c_str(),
                  "An aggregator address must have the form <address>:<port> "
                  "where address is a valid IPv4 address");
            });

  std::vector<std::thread> threads;
  threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    threads.emplace_back(
        [](void *sub) {
          zmq_msg_t msg;
          zmq_msg_init(&msg);
          ASSERT_EQ(zmq_msg_recv(&msg, sub, 0), -1);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  for (auto &thread : threads)
    thread.join();
}
