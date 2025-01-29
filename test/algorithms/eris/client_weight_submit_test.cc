#include "algorithms/eris/aggregator.pb.h"
#include "mock_client.h"
#include "mock_zmq_socket.h"
#include "zmq.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <numeric>
#include <random>
#include <vector>

static const uint32_t splits = 5;
static const uint32_t split_seed = 42;
static const uint32_t fragment_size = 10;
static constexpr size_t parameters_size = splits * fragment_size;

class ClientWeightSubmitTest : public testing::Test {
protected:
  ClientWeightSubmitTest(void) : client_{}, expected_(parameters_size) {
    std::default_random_engine rng(time(NULL));
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    client_.get_splitter().configure(parameters_size, splits, split_seed);
    client_.get_options().set_splits(splits);
    client_.get_options().set_split_seed(split_seed);

    client_.get_submit_sockets().resize(splits);

    for (size_t i = 0; i < expected_.size(); ++i)
      expected_[i] = dist(rng);

    client_.set_parameters(expected_);
  }

  std::vector<float>
  Reassemble(const std::vector<eris::WeightSubmissionRequest> &requests) {
    std::vector<eris::WeightUpdate> results;
    results.reserve(requests.size());

    for (size_t i = 0; i < requests.size(); ++i) {
      eris::WeightUpdate update;

      for (int j = 0; j < requests[i].weight_size(); ++j)
        update.add_weight(requests[i].weight(j));

      results.emplace_back(update);
    }

    return client_.get_splitter().reassemble(results);
  }

  void CheckSubmission(uint32_t round) {
    std::vector<eris::WeightSubmissionRequest> requests;
    requests.resize(splits);

    for (size_t i = 0; i < splits; ++i) {
      zmq_msg_t msg;

      zmq_msg_init(&msg);
      EXPECT_TRUE(client_.get_submit_sockets()[i]->send_dequeue(&msg));
      EXPECT_TRUE(
          requests[i].ParseFromArray(zmq_msg_data(&msg), zmq_msg_size(&msg)));
      EXPECT_EQ(requests[i].round(), round);
      zmq_msg_close(&msg);
    }

    std::vector<float> submitted = Reassemble(requests);
    EXPECT_EQ(submitted.size(), expected_.size());
    for (size_t i = 0; i < expected_.size(); ++i)
      EXPECT_NEAR(submitted[i], expected_[i],
                  5 * std::numeric_limits<float>::epsilon());
  }

  std::future<void> RegisterAggregator(uint32_t id) {
    zmq_msg_t msg;
    eris::WeightSubmissionResponse res;
    std::future<void> ret;

    zmq_msg_init_size(&msg, res.ByteSizeLong());
    {
      std::lock_guard lk(client_);
      client_.get_submit_sockets()[id] =
          std::make_unique<MockZMQSocket>(ZMQ_DEALER);
      ret = client_.get_submit_sockets()[id]->recv_enqueue(msg);
    }
    client_.notify();
    return ret;
  }

  ~ClientWeightSubmitTest(void) = default;

  void TearDown() {
    EXPECT_TRUE(client_.get_dealer().is_empty());
    EXPECT_TRUE(client_.get_subscriber().is_empty());
  }

  MockClient client_;
  std::vector<float> expected_;
};

TEST_F(ClientWeightSubmitTest, SubmitWeights) {
  srand(time(NULL));
  uint32_t round = rand() % 30;
  std::vector<std::shared_future<void>> responses;
  responses.reserve(splits);

  for (auto &sock : client_.get_submit_sockets()) {
    zmq_msg_t msg;
    eris::WeightSubmissionResponse res;
    sock = std::make_unique<MockZMQSocket>(ZMQ_DEALER);
    zmq_msg_init_size(&msg, res.ByteSizeLong());
    responses.emplace_back(sock->recv_enqueue(msg));
  }

  EXPECT_TRUE(client_.mock_submit_weights(round));
  CheckSubmission(round);

  for (std::shared_future<void> t : responses)
    t.wait();
}

TEST_F(ClientWeightSubmitTest, OneAggregatorJoinLater) {
  srand(time(NULL));
  uint32_t round = rand() % 30;
  size_t missing = rand() % splits;

  std::vector<std::shared_future<void>> responses;
  responses.reserve(splits);

  for (size_t i = 0; i < splits; ++i) {
    zmq_msg_t msg;
    eris::WeightSubmissionResponse res;

    if (i == missing)
      continue;
    client_.get_submit_sockets()[i] =
        std::make_unique<MockZMQSocket>(ZMQ_DEALER);
    zmq_msg_init_size(&msg, res.ByteSizeLong());
    responses.emplace_back(client_.get_submit_sockets()[i]->recv_enqueue(msg));
  }

  std::future<bool> submitted = std::async(
      std::launch::async,
      [&](uint32_t round) { return client_.mock_submit_weights(round); },
      round);
  std::future<void> received = RegisterAggregator(missing);

  received.wait();
  for (std::shared_future<void> t : responses)
    t.wait();
  submitted.wait();
  EXPECT_TRUE(submitted.get());
  CheckSubmission(round);
}

TEST_F(ClientWeightSubmitTest, SubmitWhileAggregatorsJoining) {
  srand(time(NULL));
  uint32_t round = rand() % 30;
  std::vector<uint32_t> missing(splits);

  std::iota(missing.begin(), missing.end(), 0);
  std::shuffle(missing.begin(), missing.end(),
               std::default_random_engine(time(NULL)));
  missing.resize(splits - 2);

  std::vector<std::shared_future<void>> responses;
  responses.reserve(splits);

  for (size_t i = 0; i < splits; ++i) {
    zmq_msg_t msg;
    eris::WeightSubmissionResponse res;

    if (std::find(missing.begin(), missing.end(), i) != missing.end())
      continue;
    client_.get_submit_sockets()[i] =
        std::make_unique<MockZMQSocket>(ZMQ_DEALER);
    zmq_msg_init_size(&msg, res.ByteSizeLong());
    responses.emplace_back(client_.get_submit_sockets()[i]->recv_enqueue(msg));
  }

  std::future<bool> submitted = std::async(
      std::launch::async,
      [&](uint32_t round) { return client_.mock_submit_weights(round); },
      round);

  for (const uint32_t i : missing)
    responses.emplace_back(RegisterAggregator(i));

  for (std::shared_future<void> t : responses)
    t.wait();
  submitted.wait();
  EXPECT_TRUE(submitted.get());
  CheckSubmission(round);
}
