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
#include <utility>
#include <vector>

using eris::InitialState;
using eris::JoinRequest;
using grpc::Status;
using grpc::StatusCode;

static constexpr std::chrono::minutes timeout = std::chrono::minutes(1);
static const int zmq_timeout = 500;
static const uint32_t min_clients = 2;
static const uint32_t rounds = 10;
static const uint32_t split_seed = 42;
static const uint32_t splits = 10;

static void validate_training_options(const InitialState &state) {
  EXPECT_EQ(state.options().min_clients(), min_clients);
  EXPECT_EQ(state.options().rounds(), rounds);
  EXPECT_EQ(state.options().split_seed(), split_seed);
  EXPECT_EQ(state.options().splits(), splits);
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
    builder.add_publish_port(0);
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
      EXPECT_NE(ctx[i], nullptr);
      subscriber[i] = zmq_socket(ctx[i], ZMQ_SUB);
      EXPECT_NE(subscriber[i], nullptr);
      EXPECT_EQ(zmq_connect(subscriber[i], get_pubsub_address().c_str()), 0);
      EXPECT_EQ(zmq_setsockopt(subscriber[i], ZMQ_SUBSCRIBE, "", 0), 0);
      EXPECT_EQ(zmq_setsockopt(subscriber[i], ZMQ_RCVTIMEO, &zmq_timeout,
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
    return "tcp://127.0.0.1:" + std::to_string(server_->get_publish_port());
  }

  static const size_t subscribers = 5;
  std::shared_ptr<ErisCoordinator> server_;
  std::unique_ptr<std::thread> server_thread_;
  void *ctx[subscribers];
  void *subscriber[subscribers];
};

TEST_F(ErisCoordinatorTest, Initialization) {
  EXPECT_NE(server_, nullptr);
  EXPECT_NE(server_thread_, nullptr);
}

TEST_F(ErisCoordinatorTest, JoinClient) {
  EXPECT_NE(server_, nullptr);
  EXPECT_NE(server_thread_, nullptr);

  JoinRequest req;
  InitialState res;

  test_join(get_rpc_address(), req, &res,
            [](Status status, InitialState *state) {
              EXPECT_TRUE(status.ok());
              validate_training_options(*state);
              EXPECT_FALSE(state->has_assigned_fragment());
              EXPECT_EQ(state->aggregators_size(), 0);
            });

  std::vector<std::thread> threads;
  threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    threads.emplace_back(
        [](void *sub) {
          zmq_msg_t msg;
          zmq_msg_init(&msg);
          EXPECT_EQ(zmq_msg_recv(&msg, sub, 0), -1);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  for (auto &thread : threads)
    thread.join();
}

TEST_F(ErisCoordinatorTest, JoinAggregator) {
  EXPECT_NE(server_, nullptr);
  EXPECT_NE(server_thread_, nullptr);

  const std::string submit_address = "127.0.0.0:50052";
  const std::string publish_address = "tcp://127.0.0.0:50052";

  InitialState res;
  JoinRequest req;
  *req.mutable_submit_address() = submit_address;
  *req.mutable_publish_address() = publish_address;

  std::vector<std::thread> threads;
  threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    threads.emplace_back(
        [&publish_address, &submit_address](void *sub) {
          zmq_msg_t msg;
          FragmentInfo info;

          zmq_msg_init(&msg);
          int size = zmq_msg_recv(&msg, sub, 0);
          EXPECT_NE(size, -1);
          EXPECT_TRUE(info.ParseFromArray(zmq_msg_data(&msg), size));
          EXPECT_EQ(info.publish_address(), publish_address);
          EXPECT_EQ(info.submit_address(), submit_address);
          EXPECT_LT(info.id(), splits);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  test_join(
      get_rpc_address(), req, &res,
      [&publish_address, &submit_address](Status status, InitialState *state) {
        EXPECT_TRUE(status.ok());
        validate_training_options(*state);
        EXPECT_TRUE(state->has_assigned_fragment());
        EXPECT_EQ(state->aggregators_size(), 1);
        EXPECT_GE(state->assigned_fragment(), 0);
        EXPECT_LT(state->assigned_fragment(), splits);

        bool found = false;

        for (auto aggr : state->aggregators())
          if (aggr.publish_address() == publish_address &&
              aggr.submit_address() == submit_address)
            found = true;

        EXPECT_TRUE(found);
      });

  for (auto &thread : threads)
    thread.join();
}

TEST_F(ErisCoordinatorTest, JoinTooManyAggregators) {
  EXPECT_NE(server_, nullptr);
  EXPECT_NE(server_thread_, nullptr);

  std::set<std::pair<std::string, std::string>> aggregators;

  const uint16_t base_port = 50052;
  for (uint16_t i = 0; i < splits; ++i)
    aggregators.insert(
        std::make_pair("127.0.0.0:" + std::to_string(base_port + i),
                       "tcp://127.0.0.1:" + std::to_string(base_port + i)));

  std::vector<std::thread> subscriber_threads;
  subscriber_threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    subscriber_threads.emplace_back(
        [&aggregators](void *sub) {
          zmq_msg_t msg;
          FragmentInfo info;

          std::set<std::pair<std::string, std::string>> addresses;
          std::unordered_set<uint32_t> ids;

          for (size_t i = 0; i < aggregators.size(); ++i) {
            zmq_msg_init(&msg);
            int size = zmq_msg_recv(&msg, sub, 0);
            EXPECT_NE(size, -1);
            EXPECT_TRUE(info.ParseFromArray(zmq_msg_data(&msg), size));
            std::pair<std::string, std::string> aggregator{
                info.submit_address(), info.publish_address()};
            EXPECT_NE(aggregators.find(aggregator), aggregators.end());
            EXPECT_LT(info.id(), splits);
            addresses.insert(aggregator);
            ids.insert(info.id());
            zmq_msg_close(&msg);
          }

          EXPECT_EQ(addresses.size(), aggregators.size());
          EXPECT_EQ(ids.size(), splits);

          zmq_msg_init(&msg);
          EXPECT_EQ(zmq_msg_recv(&msg, sub, 0), -1);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  std::vector<std::thread> threads;
  threads.reserve(aggregators.size());

  for (const std::pair<std::string, std::string> &aggregator : aggregators)
    threads.emplace_back(
        [this](const std::pair<std::string, std::string> &aggregator) {
          InitialState res;
          JoinRequest req;
          req.set_publish_address(aggregator.second);
          req.set_submit_address(aggregator.first);

          test_join(get_rpc_address(), req, &res,
                    [&aggregator](Status status, InitialState *state) {
                      EXPECT_TRUE(status.ok());
                      validate_training_options(*state);
                      EXPECT_TRUE(state->has_assigned_fragment());
                      EXPECT_GE(state->assigned_fragment(), 0);
                      EXPECT_LT(state->assigned_fragment(), splits);
                      bool found = false;

                      for (auto aggr : state->aggregators())
                        if (aggr.publish_address() == aggregator.second &&
                            aggr.submit_address() == aggregator.first)
                          found = true;

                      EXPECT_TRUE(found);
                    });
        },
        aggregator);

  for (auto &thread : threads)
    thread.join();

  InitialState res;
  JoinRequest req;
  req.set_submit_address("127.0.0.0:" +
                         std::to_string(base_port + aggregators.size() + 1));
  req.set_publish_address("tcp://127.0.0.0:" +
                          std::to_string(base_port + aggregators.size() + 1));

  test_join(get_rpc_address(), req, &res,
            [&aggregators](Status status, InitialState *state) {
              EXPECT_TRUE(status.ok());
              validate_training_options(*state);
              EXPECT_FALSE(state->has_assigned_fragment());

              EXPECT_EQ(aggregators.size(), state->aggregators_size());

              for (auto aggr : state->aggregators()) {
                std::pair<std::string, std::string> aggregator{
                    aggr.submit_address(), aggr.publish_address()};
                if (aggregators.find(aggregator) == aggregators.end())
                  FAIL();
              }
            });

  for (auto &thread : subscriber_threads)
    thread.join();
}

TEST_F(ErisCoordinatorTest, MissingPublishAddress) {
  EXPECT_NE(server_, nullptr);
  EXPECT_NE(server_thread_, nullptr);

  JoinRequest req;
  InitialState res;
  req.set_submit_address("127.0.0.1:50051");

  test_join(get_rpc_address(), req, &res,
            [](Status status, InitialState *state) {
              EXPECT_FALSE(status.ok());
              EXPECT_EQ(status.error_code(), StatusCode::INVALID_ARGUMENT);
              EXPECT_STREQ(status.error_message().c_str(),
                           "Missing model updates publishing address");
            });

  std::vector<std::thread> threads;
  threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    threads.emplace_back(
        [](void *sub) {
          zmq_msg_t msg;
          zmq_msg_init(&msg);
          EXPECT_EQ(zmq_msg_recv(&msg, sub, 0), -1);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  for (auto &thread : threads)
    thread.join();
}

TEST_F(ErisCoordinatorTest, MissingSubmitAddress) {
  EXPECT_NE(server_, nullptr);
  EXPECT_NE(server_thread_, nullptr);

  JoinRequest req;
  InitialState res;
  req.set_publish_address("127.0.0.1:50051");

  test_join(get_rpc_address(), req, &res,
            [](Status status, InitialState *state) {
              EXPECT_FALSE(status.ok());
              EXPECT_EQ(status.error_code(), StatusCode::INVALID_ARGUMENT);
              EXPECT_STREQ(status.error_message().c_str(),
                           "Missing weight submission address");
            });

  std::vector<std::thread> threads;
  threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    threads.emplace_back(
        [](void *sub) {
          zmq_msg_t msg;
          zmq_msg_init(&msg);
          EXPECT_EQ(zmq_msg_recv(&msg, sub, 0), -1);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  for (auto &thread : threads)
    thread.join();
}

TEST_F(ErisCoordinatorTest, InvalidPublishAddress) {
  EXPECT_NE(server_, nullptr);
  EXPECT_NE(server_thread_, nullptr);

  JoinRequest req;
  InitialState res;
  req.set_publish_address("Some random string");
  req.set_submit_address("127.0.0.1:50051");

  test_join(get_rpc_address(), req, &res,
            [](Status status, InitialState *state) {
              EXPECT_FALSE(status.ok());
              EXPECT_EQ(status.error_code(), StatusCode::INVALID_ARGUMENT);
              EXPECT_STREQ(status.error_message().c_str(),
                           "A model updates publishing address must have the "
                           "form tcp://<address>:<port>"
                           "where address is a valid IPv4 address");
            });

  std::vector<std::thread> threads;
  threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    threads.emplace_back(
        [](void *sub) {
          zmq_msg_t msg;
          zmq_msg_init(&msg);
          EXPECT_EQ(zmq_msg_recv(&msg, sub, 0), -1);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  for (auto &thread : threads)
    thread.join();
}

TEST_F(ErisCoordinatorTest, InvalidSubmitAddress) {
  EXPECT_NE(server_, nullptr);
  EXPECT_NE(server_thread_, nullptr);

  JoinRequest req;
  InitialState res;
  req.set_publish_address("127.0.0.1:5000");
  req.set_submit_address("Some random string");

  test_join(
      get_rpc_address(), req, &res, [](Status status, InitialState *state) {
        EXPECT_FALSE(status.ok());
        EXPECT_EQ(status.error_code(), StatusCode::INVALID_ARGUMENT);
        EXPECT_STREQ(
            status.error_message().c_str(),
            "A weight submission address must have the form <address>:<port>"
            "where address is a valid IPv4 address");
      });

  std::vector<std::thread> threads;
  threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    threads.emplace_back(
        [](void *sub) {
          zmq_msg_t msg;
          zmq_msg_init(&msg);
          EXPECT_EQ(zmq_msg_recv(&msg, sub, 0), -1);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  for (auto &thread : threads)
    thread.join();
}
