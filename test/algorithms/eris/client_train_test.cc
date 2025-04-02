#include <algorithms/eris/common.pb.h>
#include <algorithms/eris/config.h>
#include <algorithms/eris/coordinator.pb.h>
#include <cstdint>
#include <ctime>
#include <future>
#include <gtest/gtest.h>
#include <mock_client.h>
#include <mock_zmq_socket.h>
#include <string>
#include <utility>
#include <zmq.h>

class TrainSocket : public MockZMQSocket {
public:
  explicit TrainSocket(int type) : MockZMQSocket{type} {}
  ~TrainSocket(void) = default;
};

class TrainClient : public ErisClient<std::vector<float>, TrainSocket> {
public:
  explicit TrainClient(void)
      : ErisClient<std::vector<float>,
                   TrainSocket>{"tcp://127.0.0.1:" +
                                    std::to_string(DEFAULT_ERIS_ROUTER_PORT),
                                "tcp://127.0.0.1:" +
                                    std::to_string(DEFAULT_ERIS_PUBLISH_PORT)},
        fit_calls_{0}, evaluate_calls_{0}, parameters_(100) {}

  std::vector<float> get_parameters(void) { return parameters_; }

  void set_parameters(const std::vector<float> &parameters) {
    parameters_ = parameters;
  }

  fit_result fit(void) {
    ++fit_calls_;
    return std::make_pair(get_parameters(), 1);
  }

  void evaluate(void) { ++evaluate_calls_; };

  inline uint32_t get_fit_calls(void) const { return fit_calls_; }
  inline uint32_t get_evaluate_calls(void) const { return evaluate_calls_; }

private:
  uint32_t fit_calls_;
  uint32_t evaluate_calls_;

  std::vector<float> parameters_;
};

class ClientTrainFailedTest : public testing::Test {
protected:
  ClientTrainFailedTest(void) : client_{}, state_{} {}
  ~ClientTrainFailedTest(void) = default;

  void SetUp(void) {
    eris::TrainingOptions options;
    uint32_t splits = 5;

    options.set_min_clients(1);
    options.set_rounds(2);
    options.set_split_seed(10);
    options.set_splits(splits);
    *state_.mutable_options() = options;

    for (uint32_t i = 0; i < splits; ++i) {
      eris::FragmentInfo fragment;
      fragment.set_id(i);
      fragment.set_submit_address("tcp://127.0.0.1:" +
                                  std::to_string(DEFAULT_ERIS_ROUTER_PORT + i));
      fragment.set_publish_address(
          "tcp://127.0.0.1:" + std::to_string(DEFAULT_ERIS_PUBLISH_PORT + i));

      *state_.add_aggregators() = fragment;
    }
    client_.set_parameters(std::vector<float>(100));
  }

  void TearDown() {
    EXPECT_TRUE(client_.get_dealer().is_empty());
    EXPECT_TRUE(client_.get_subscriber().is_empty());
  }

  std::future<void> AddResponse(eris::StateResponse &res) {
    zmq_msg_t msg;

    zmq_msg_init_size(&msg, res.ByteSizeLong());
    res.SerializeToArray(zmq_msg_data(&msg), res.ByteSizeLong());
    return client_.get_dealer().recv_enqueue(std::move(msg));
  }

  eris::StateRequest GetRequest(void) {
    zmq_msg_t msg;
    eris::StateRequest res;

    zmq_msg_init(&msg);
    EXPECT_TRUE(client_.get_dealer().send_dequeue(&msg));
    EXPECT_TRUE(res.ParseFromArray(zmq_msg_data(&msg), zmq_msg_size(&msg)));
    zmq_msg_close(&msg);

    return res;
  }

  MockClient client_;
  eris::State state_;
};

TEST_F(ClientTrainFailedTest, JoinClientFailed) {
  eris::StateResponse res;
  eris::Error error;

  error.set_code(eris::ErrorCode::INVALID_ARGUMENT);
  error.set_msg("joining is not allowed");
  *res.mutable_error() = error;

  std::future<void> response_received = AddResponse(res);

  EXPECT_FALSE(client_.join());
  EXPECT_FALSE(client_.train());

  eris::StateRequest req = GetRequest();
  EXPECT_TRUE(req.has_join());
  EXPECT_FALSE(req.join().has_publish_address());
  EXPECT_FALSE(req.join().has_submit_address());

  response_received.wait();
  EXPECT_FALSE(client_.is_aggregator());
}

TEST_F(ClientTrainFailedTest, JoinAggregatorFailed) {
  eris::StateResponse res;
  eris::Error error;

  error.set_code(eris::ErrorCode::INVALID_ARGUMENT);
  error.set_msg("joining is not allowed");
  *res.mutable_error() = error;

  std::future<void> response_received = AddResponse(res);
  EXPECT_TRUE(client_.set_aggregator_config("127.0.0.1", 10, 20));
  EXPECT_FALSE(client_.join());
  EXPECT_FALSE(client_.train());

  eris::StateRequest req = GetRequest();
  EXPECT_TRUE(req.has_join());
  EXPECT_TRUE(req.join().has_publish_address());
  EXPECT_TRUE(req.join().has_submit_address());
  EXPECT_EQ(req.join().submit_address(), "tcp://127.0.0.1:10");
  EXPECT_EQ(req.join().publish_address(), "tcp://127.0.0.1:20");

  response_received.wait();
  EXPECT_FALSE(client_.is_aggregator());
}
