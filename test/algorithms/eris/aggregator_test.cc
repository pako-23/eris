#include "algorithms/eris/aggregator.grpc.pb.h"
#include "algorithms/eris/aggregator.h"
#include "algorithms/eris/builder.h"
#include "algorithms/eris/common.pb.h"
#include "zmq.h"
#include <cstddef>
#include <cstdint>
#include <grpcpp/create_channel.h>
#include <grpcpp/support/status.h>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <vector>

using grpc::Status;

static const int zmq_timeout = 500;
static constexpr std::chrono::minutes timeout = std::chrono::minutes(1);
static const uint32_t min_clients = 5;
static const size_t fragment_size = 10;
static const uint32_t fragment_id = 1;

static void test_submit(const std::string &aggregator, uint32_t round,
                        std::vector<float>::const_iterator begin,
                        std::vector<float>::const_iterator end,
                        std::function<void(Status)> check) {
  FragmentWeights req;
  eris::Empty res;

  req.set_round(round);
  for (; begin != end; ++begin)
    req.add_weight(*begin);

  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(aggregator, grpc::InsecureChannelCredentials());
  std::unique_ptr<eris::Aggregator::Stub> stub =
      eris::Aggregator::NewStub(channel);

  grpc::ClientContext ctx;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;

  stub->async()->SubmitWeights(&ctx, &req, &res,
                               [&check, &mu, &done, &cv, res](Status s) {
                                 check(s);
                                 std::lock_guard<std::mutex> lk(mu);

                                 done = true;
                                 cv.notify_one();
                               });

  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [&done] { return done; });
}

class ErisAggregatorTest : public testing::Test {
protected:
  ErisAggregatorTest(void) : server_{nullptr}, server_thread_{nullptr} {
    ErisAggregatorBuilder builder(fragment_id, fragment_size);

    builder.add_rpc_port(0);
    builder.add_publish_port(0);
    builder.add_min_clients(min_clients);

    server_ = std::make_shared<ErisAggregator>(builder);
    server_thread_ = std::make_unique<std::thread>(
        [](std::shared_ptr<ErisAggregator> aggregator) { aggregator->start(); },
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

  ~ErisAggregatorTest(void) {
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
    generate_weights();
  }

  void generate_weights(void) {
    std::default_random_engine rng(time(NULL));
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    weights.clear();
    weights.reserve(min_clients);
    expected.clear();
    expected.resize(fragment_size, 0.0);

    for (uint32_t i = 0; i < min_clients; ++i) {
      std::vector<float> client_weights;
      client_weights.reserve(fragment_size);

      for (size_t j = 0; j < fragment_size; ++j) {
        client_weights.emplace_back(dist(rng));
        expected[j] += client_weights.back();
      }

      weights.emplace_back(client_weights);
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
  std::shared_ptr<ErisAggregator> server_;
  std::unique_ptr<std::thread> server_thread_;
  void *ctx[subscribers];
  void *subscriber[subscribers];
  std::vector<std::vector<float>> weights;
  std::vector<float> expected;
};

TEST_F(ErisAggregatorTest, Initialization) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);
}

TEST_F(ErisAggregatorTest, WeightSubmission) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);

  std::vector<std::thread> threads;
  threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    threads.emplace_back(
        [this](void *sub) {
          zmq_msg_t msg;
          WeightUpdate update;

          zmq_msg_init(&msg);
          int size = zmq_msg_recv(&msg, sub, 0);
          ASSERT_NE(size, -1);
          ASSERT_TRUE(update.ParseFromArray(zmq_msg_data(&msg), size));
          ASSERT_EQ(update.round(), 0);
          ASSERT_EQ(update.contributors(), min_clients);
          ASSERT_EQ(update.weight_size(), fragment_size);

          for (int i = 0; i < update.weight_size(); ++i)
            EXPECT_NEAR(update.weight(i), expected[i],
                        5 * std::numeric_limits<float>::epsilon());
          zmq_msg_close(&msg);

          zmq_msg_init(&msg);
          ASSERT_EQ(zmq_msg_recv(&msg, sub, 0), -1);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  for (size_t i = 0; i < min_clients; ++i)
    test_submit(get_rpc_address(), 0, weights[i].cbegin(), weights[i].cend(),
                [](Status s) { ASSERT_TRUE(s.ok()); });

  for (auto &thread : threads)
    thread.join();
}

TEST_F(ErisAggregatorTest, InvalidRound) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);

  std::vector<std::thread> subscriber_threads;
  subscriber_threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    subscriber_threads.emplace_back(
        [](void *sub) {
          zmq_msg_t msg;
          zmq_msg_init(&msg);
          ASSERT_EQ(zmq_msg_recv(&msg, sub, 0), -1);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  test_submit(get_rpc_address(), 1, weights[0].cbegin(), weights[0].cend(),
              [](Status s) {
                ASSERT_FALSE(s.ok());
                ASSERT_EQ(s.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
                ASSERT_STREQ(s.error_message().c_str(), "Wrong round number");
              });

  for (auto &thread : subscriber_threads)
    thread.join();
}

TEST_F(ErisAggregatorTest, InvalidFragmentSize) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);

  std::vector<std::thread> subscriber_threads;
  subscriber_threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    subscriber_threads.emplace_back(
        [](void *sub) {
          zmq_msg_t msg;
          zmq_msg_init(&msg);
          ASSERT_EQ(zmq_msg_recv(&msg, sub, 0), -1);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  test_submit(get_rpc_address(), 0, weights[0].cbegin(), weights[0].cend() - 1,
              [](Status s) {
                ASSERT_FALSE(s.ok());
                ASSERT_EQ(s.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
                ASSERT_STREQ(s.error_message().c_str(), "Wrong fragment size");
              });

  for (auto &thread : subscriber_threads)
    thread.join();
}

TEST_F(ErisAggregatorTest, MultipleRounds) {
  ASSERT_NE(server_, nullptr);
  ASSERT_NE(server_thread_, nullptr);

  const int rounds = 3;

  std::vector<std::thread> threads;
  threads.reserve(subscribers);

  for (size_t i = 0; i < subscribers; ++i)
    threads.emplace_back(
        [this](void *sub) {
          zmq_msg_t msg;
          WeightUpdate update;

          for (int i = 0; i < rounds; ++i) {
            zmq_msg_init(&msg);
            int size = zmq_msg_recv(&msg, sub, 0);
            ASSERT_NE(size, -1);
            ASSERT_TRUE(update.ParseFromArray(zmq_msg_data(&msg), size));
            ASSERT_EQ(update.round(), i);
            ASSERT_EQ(update.contributors(), min_clients);
            ASSERT_EQ(update.weight_size(), fragment_size);

            for (int i = 0; i < update.weight_size(); ++i)
              EXPECT_NEAR(update.weight(i), expected[i],
                          5 * std::numeric_limits<float>::epsilon());
            zmq_msg_close(&msg);
          }

          zmq_msg_init(&msg);
          EXPECT_EQ(zmq_msg_recv(&msg, sub, 0), -1);
          zmq_msg_close(&msg);
        },
        subscriber[i]);

  for (int round = 0; round < rounds; ++round) {
    for (size_t i = 0; i < min_clients; ++i)
      test_submit(get_rpc_address(), round, weights[i].cbegin(),
                  weights[i].cend(), [](Status s) { EXPECT_TRUE(s.ok()); });

    generate_weights();
  }

  for (auto &thread : threads)
    thread.join();
}
