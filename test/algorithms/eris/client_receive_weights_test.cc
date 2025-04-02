#include "algorithms/eris/aggregator.pb.h"
#include "mock_client.h"
#include "mock_zmq_socket.h"
#include "zmq.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <sys/types.h>

static const uint32_t splits = 5;
static const uint32_t split_seed = 42;
static const uint32_t fragment_size = 10;
static constexpr size_t parameters_size = splits * fragment_size;

class ClientReceiveWeightsTest : public testing::Test {
protected:
  ClientReceiveWeightsTest(void) : client_{} {
    client_.get_splitter().configure(parameters_size, splits, split_seed);
    client_.get_options().set_splits(splits);
    client_.get_options().set_split_seed(split_seed);

    client_.get_publish_sockets().resize(splits);
    for (auto &sock : client_.get_publish_sockets())
      sock = std::make_unique<MockZMQSocket>(ZMQ_SUB);
  }

  std::vector<float> GenerateRandomVector(void) {
    std::vector<float> vec(parameters_size);
    std::default_random_engine rng(time(NULL));
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (size_t i = 0; i < vec.size(); ++i)
      vec[i] = dist(rng);

    return vec;
  }

  std::vector<eris::WeightUpdate>
  GenerateUpdates(const std::vector<float> &parameters, uint32_t round) {
    std::vector<eris::WeightSubmissionRequest> requests =
        client_.get_splitter().split(parameters.begin(), parameters.end(), 1,
                                     round);
    std::vector<eris::WeightUpdate> results;
    results.reserve(requests.size());

    for (size_t i = 0; i < requests.size(); ++i) {
      eris::WeightUpdate update;

      update.set_round(round);
      for (int j = 0; j < requests[i].weight_size(); ++j)
        update.add_weight(requests[i].weight(j));

      results.emplace_back(update);
    }

    return results;
  }

  ~ClientReceiveWeightsTest(void) = default;

  void TearDown() {
    EXPECT_TRUE(client_.get_dealer().is_empty());
    EXPECT_TRUE(client_.get_subscriber().is_empty());
  }

  MockClient client_;
};

TEST_F(ClientReceiveWeightsTest, ReceiveWeights) {
  uint32_t round = 0;
  std::vector<float> expected = GenerateRandomVector();
  std::vector<eris::WeightUpdate> updates = GenerateUpdates(expected, round);
  std::vector<std::shared_future<void>> received;
  received.reserve(updates.size());
  std::vector<float> parameters(expected.size());

  for (size_t i = 0; i < updates.size(); ++i) {
    zmq_msg_t msg;

    zmq_msg_init_size(&msg, updates[i].ByteSizeLong());
    updates[i].SerializeToArray(zmq_msg_data(&msg), updates[i].ByteSizeLong());
    received.emplace_back(client_.get_publish_sockets()[i]->recv_enqueue(msg));
  }

  EXPECT_TRUE(client_.mock_receive_weights(&round, parameters));
  EXPECT_EQ(round, 0);
  std::vector<float> weights = client_.get_parameters();
  EXPECT_EQ(weights.size(), expected.size());
  for (size_t i = 0; i < weights.size(); ++i)
    EXPECT_NEAR(weights[i], expected[i],
                5 * std::numeric_limits<float>::epsilon());

  for (std::shared_future<void> &t : received)
    t.wait();
}

TEST_F(ClientReceiveWeightsTest, ReceiveOlderWeights) {
  uint32_t round = 1;

  std::vector<float> expected = GenerateRandomVector();
  std::vector<std::vector<eris::WeightUpdate>> updates{
      GenerateUpdates(GenerateRandomVector(), 0), GenerateUpdates(expected, 1)};
  std::vector<float> parameters(expected.size());
  std::vector<std::shared_future<void>> received;
  received.reserve(splits * 2);

  for (size_t i = 0; i < updates.size(); ++i) {
    for (size_t j = 0; j < updates[i].size(); ++j) {
      zmq_msg_t msg;

      zmq_msg_init_size(&msg, updates[i][j].ByteSizeLong());
      updates[i][j].SerializeToArray(zmq_msg_data(&msg),
                                     updates[i][j].ByteSizeLong());
      received.emplace_back(
          client_.get_publish_sockets()[j]->recv_enqueue(msg));
    }
  }

  EXPECT_TRUE(client_.mock_receive_weights(&round, parameters));
  EXPECT_EQ(round, 1);
  std::vector<float> weights = client_.get_parameters();
  EXPECT_EQ(weights.size(), expected.size());
  for (size_t i = 0; i < weights.size(); ++i)
    EXPECT_NEAR(weights[i], expected[i],
                5 * std::numeric_limits<float>::epsilon());

  for (std::shared_future<void> &t : received)
    t.wait();
}

TEST_F(ClientReceiveWeightsTest, ReceiveNewerWeights) {
  uint32_t round = 0;

  std::vector<float> expected = GenerateRandomVector();
  std::vector<std::vector<eris::WeightUpdate>> updates{
      GenerateUpdates(GenerateRandomVector(), 0), GenerateUpdates(expected, 1)};
  std::vector<float> parameters(expected.size());
  std::vector<std::shared_future<void>> received;
  received.reserve(splits * 2 - 1);

  srand(time(NULL));

  for (size_t i = 0; i < updates.size(); ++i) {
    size_t missing = rand() % updates[i].size();

    for (size_t j = 0; j < updates[i].size(); ++j) {
      zmq_msg_t msg;

      if (j == missing && i != updates.size() - 1)
        continue;

      zmq_msg_init_size(&msg, updates[i][j].ByteSizeLong());
      updates[i][j].SerializeToArray(zmq_msg_data(&msg),
                                     updates[i][j].ByteSizeLong());
      received.emplace_back(
          client_.get_publish_sockets()[j]->recv_enqueue(msg));
    }
  }

  EXPECT_TRUE(client_.mock_receive_weights(&round, parameters));
  EXPECT_EQ(round, 1);
  std::vector<float> weights = client_.get_parameters();
  EXPECT_EQ(weights.size(), expected.size());
  for (size_t i = 0; i < weights.size(); ++i)
    EXPECT_NEAR(weights[i], expected[i],
                5 * std::numeric_limits<float>::epsilon());

  for (std::shared_future<void> &t : received)
    t.wait();
}

TEST_F(ClientReceiveWeightsTest, ReceiveNewerWeightsMultipleTimes) {
  uint32_t round = 0;

  std::vector<float> expected = GenerateRandomVector();
  std::vector<std::vector<eris::WeightUpdate>> updates{
      GenerateUpdates(GenerateRandomVector(), 0),
      GenerateUpdates(GenerateRandomVector(), 1), GenerateUpdates(expected, 2)};
  std::vector<float> parameters(expected.size());
  std::vector<std::shared_future<void>> received;
  received.reserve(splits * 3 - 2);
  srand(time(NULL));

  for (size_t i = 0; i < updates.size(); ++i) {
    size_t missing = rand() % updates[i].size();

    for (size_t j = 0; j < updates[i].size(); ++j) {
      zmq_msg_t msg;

      if (j == missing && i != updates.size() - 1)
        continue;

      zmq_msg_init_size(&msg, updates[i][j].ByteSizeLong());
      updates[i][j].SerializeToArray(zmq_msg_data(&msg),
                                     updates[i][j].ByteSizeLong());
      received.emplace_back(
          client_.get_publish_sockets()[j]->recv_enqueue(msg));
    }
  }

  EXPECT_TRUE(client_.mock_receive_weights(&round, parameters));
  EXPECT_EQ(round, 2);
  std::vector<float> weights = client_.get_parameters();
  EXPECT_EQ(weights.size(), expected.size());
  for (size_t i = 0; i < weights.size(); ++i)
    EXPECT_NEAR(weights[i], expected[i],
                5 * std::numeric_limits<float>::epsilon());

  for (std::shared_future<void> &t : received)
    t.wait();
}
