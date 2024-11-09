#include "algorithms/eris/aggregator.h"
#include "algorithms/eris/aggregator.pb.h"
#include "algorithms/eris/common.pb.h"
#include "mock_zmq_socket.h"
#include "zmq.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <vector>

static const uint32_t min_clients = 5;
static const size_t fragment_size = 10;
static const uint16_t publish_port = 10;
static const uint16_t router_port = 20;
static const std::string address = "127.0.0.1";

class ErisAggregatorTest : public testing::Test {
protected:
  ErisAggregatorTest(void) : server_{nullptr}, server_thread_{} {
    ErisServiceConfig config;
    config.set_router_port(router_port);
    config.set_publish_port(publish_port);
    config.set_router_address(address);
    config.set_publish_address(address);

    server_ = std::make_shared<ErisAggregator<MockZMQSocket>>(config);
    EXPECT_NE(server_, nullptr);
    server_->configure(10, min_clients);

    std::promise<void> started;
    start_ready_ = started.get_future();

    server_thread_ = std::thread(
        [](std::shared_ptr<ErisAggregator<MockZMQSocket>> aggregator,
           std::promise<void> started) {
          aggregator->start(std::move(started));
        },
        server_, std::move(started));
  }

  ~ErisAggregatorTest(void) {
    start_ready_.wait();
    server_->stop();
    server_thread_.join();
  }

  void SetUp(void) override { GenerateWeights(); }

  void GenerateWeights(void) {
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

  void WeightSubmit(std::vector<float>::const_iterator begin,
                    std::vector<float>::const_iterator end, uint32_t round,
                    eris::WeightSubmissionResponse *res) {
    eris::WeightSubmissionRequest req;
    zmq_msg_t msg, identity;
    char id[5];

    for (int i = 0; i < 5; ++i)
      id[i] = '0' + rand() % 10;

    zmq_msg_init_size(&identity, 5);
    memcpy(zmq_msg_data(&identity), id, 5);
    std::future<void> identity_recv{
        GetRouter().recv_enqueue(std::move(identity))};

    req.set_round(round);
    for (; begin != end; ++begin)
      req.add_weight(*begin);

    zmq_msg_init_size(&msg, req.ByteSizeLong());
    req.SerializeToArray(zmq_msg_data(&msg), req.ByteSizeLong());
    std::future<void> msg_recv{GetRouter().recv_enqueue(std::move(msg))};

    identity_recv.wait();
    msg_recv.wait();

    zmq_msg_init(&identity);
    EXPECT_TRUE(GetRouter().send_dequeue(&identity));
    EXPECT_EQ(zmq_msg_size(&identity), 5);
    for (int i = 0; i < 5; ++i)
      EXPECT_EQ(static_cast<char *>(zmq_msg_data(&identity))[i], id[i]);
    zmq_msg_close(&identity);

    zmq_msg_init(&msg);
    EXPECT_TRUE(GetRouter().send_dequeue(&msg));
    res->ParseFromArray(zmq_msg_data(&msg), zmq_msg_size(&msg));
    zmq_msg_close(&msg);
  }

  inline MockZMQSocket &GetRouter(void) {
    return server_->service_.get_router();
  }
  inline MockZMQSocket &GetPublisher(void) {
    return server_->service_.get_publisher();
  }

  std::shared_ptr<ErisAggregator<MockZMQSocket>> server_;
  std::thread server_thread_;
  std::future<void> start_ready_;

  std::vector<std::vector<float>> weights;
  std::vector<float> expected;
};

TEST_F(ErisAggregatorTest, Initialization) {
  std::string pull = "tcp://" + address + ":" + std::to_string(router_port);
  std::string publisher =
      "tcp://" + address + ":" + std::to_string(publish_port);
  EXPECT_STREQ(GetRouter().get_endpoint().c_str(), pull.c_str());
  EXPECT_STREQ(GetPublisher().get_endpoint().c_str(), publisher.c_str());
}

TEST_F(ErisAggregatorTest, WeightSubmission) {
  zmq_msg_t msg;
  eris::WeightUpdate update;

  for (size_t i = 0; i < min_clients; ++i) {
    eris::WeightSubmissionResponse res;
    WeightSubmit(weights[i].cbegin(), weights[i].cend(), 0, &res);
    EXPECT_FALSE(res.has_error());
  }

  zmq_msg_init(&msg);
  EXPECT_TRUE(GetPublisher().send_dequeue(&msg));
  EXPECT_TRUE(update.ParseFromArray(zmq_msg_data(&msg), zmq_msg_size(&msg)));
  EXPECT_EQ(update.round(), 0);
  EXPECT_EQ(update.contributors(), min_clients);
  EXPECT_EQ(update.weight_size(), fragment_size);
  for (int i = 0; i < update.weight_size(); ++i)
    EXPECT_NEAR(update.weight(i), expected[i],
                5 * std::numeric_limits<float>::epsilon());
  zmq_msg_close(&msg);

  zmq_msg_init(&msg);
  EXPECT_FALSE(GetPublisher().send_dequeue(&msg, 100));
  zmq_msg_close(&msg);
}

TEST_F(ErisAggregatorTest, InvalidRound) {
  zmq_msg_t msg;
  eris::WeightSubmissionResponse res;

  WeightSubmit(weights[0].cbegin(), weights[0].cend(), 1, &res);
  EXPECT_TRUE(res.has_error());
  EXPECT_EQ(res.error().code(), eris::ErrorCode::INVALID_ARGUMENT);
  EXPECT_STREQ(res.error().msg().c_str(), "Wrong round number");

  zmq_msg_init(&msg);
  EXPECT_FALSE(GetPublisher().send_dequeue(&msg, 100));
  zmq_msg_close(&msg);
}

TEST_F(ErisAggregatorTest, InvalidFragmentSize) {
  zmq_msg_t msg;
  eris::WeightSubmissionResponse res;

  WeightSubmit(weights[0].cbegin(), weights[0].cend() - 1, 0, &res);
  EXPECT_TRUE(res.has_error());
  EXPECT_EQ(res.error().code(), eris::ErrorCode::INVALID_ARGUMENT);
  EXPECT_STREQ(res.error().msg().c_str(), "Wrong fragment size");

  zmq_msg_init(&msg);
  EXPECT_FALSE(GetPublisher().send_dequeue(&msg, 100));
  zmq_msg_close(&msg);
}

TEST_F(ErisAggregatorTest, MultipleRounds) {
  const int rounds = 3;

  for (int round = 0; round < rounds; ++round) {
    zmq_msg_t msg;
    eris::WeightUpdate update;

    for (size_t i = 0; i < min_clients; ++i) {
      eris::WeightSubmissionResponse res;
      WeightSubmit(weights[i].cbegin(), weights[i].cend(), round, &res);
      EXPECT_FALSE(res.has_error());
    }

    zmq_msg_init(&msg);
    EXPECT_TRUE(GetPublisher().send_dequeue(&msg));
    EXPECT_TRUE(update.ParseFromArray(zmq_msg_data(&msg), zmq_msg_size(&msg)));
    EXPECT_EQ(update.round(), round);
    EXPECT_EQ(update.contributors(), min_clients);
    EXPECT_EQ(update.weight_size(), fragment_size);
    for (int i = 0; i < update.weight_size(); ++i)
      EXPECT_NEAR(update.weight(i), expected[i],
                  5 * std::numeric_limits<float>::epsilon());
    zmq_msg_close(&msg);

    zmq_msg_init(&msg);
    EXPECT_FALSE(GetPublisher().send_dequeue(&msg, 100));
    zmq_msg_close(&msg);

    GenerateWeights();
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  google::protobuf::ShutdownProtobufLibrary();
  return ret;
}
