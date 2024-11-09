#include "algorithms/eris/common.pb.h"
#include "algorithms/eris/config.h"
#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "mock_zmq_socket.h"
#include "zmq.h"
#include <cstdint>
#include <cstdlib>
#include <future>
#include <gtest/gtest.h>
#include <string>
#include <thread>

static const uint32_t min_clients = 2;
static const uint32_t rounds = 10;
static const uint32_t split_seed = 42;
static const uint32_t splits = 10;
static const std::string address = "127.0.0.1";
static const uint16_t publish_port = 10;
static const uint16_t router_port = 20;

class ErisCoordinatorTest : public testing::Test {
protected:
  ErisCoordinatorTest(void) : server_{nullptr}, server_thread_{} {
    ErisCoordinatorConfig config;
    config.set_min_clients(min_clients);
    config.set_rounds(rounds);
    config.set_router_address(address);
    config.set_router_port(router_port);
    config.set_publish_address(address);
    config.set_publish_port(publish_port);
    config.set_split_seed(split_seed);
    config.set_splits(splits);

    server_ = std::make_shared<ErisCoordinator<MockZMQSocket>>(config);
    EXPECT_NE(server_, nullptr);

    std::promise<void> started;
    start_ready_ = started.get_future();

    server_thread_ = std::thread(
        [](std::shared_ptr<ErisCoordinator<MockZMQSocket>> coordinator,
           std::promise<void> started) {
          coordinator->start(std::move(started));
        },
        server_, std::move(started));
  }

  ~ErisCoordinatorTest(void) {
    start_ready_.wait();
    server_->stop();
    server_thread_.join();
  }

  void TearDown() {
    EXPECT_TRUE(GetPublisher().is_empty());
    EXPECT_TRUE(GetRouter().is_empty());
  }

  void TestJoin(const eris::JoinRequest &join_req, eris::StateResponse *res) {
    zmq_msg_t msg, identity;
    eris::StateRequest req;
    char id[5];

    *req.mutable_join() = join_req;
    for (int i = 0; i < 5; ++i)
      id[i] = '0' + rand() % 10;

    zmq_msg_init_size(&identity, 5);
    memcpy(zmq_msg_data(&identity), id, 5);
    std::future<void> identity_recv{
        GetRouter().recv_enqueue(std::move(identity))};

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

  void ValidateTrainingOptions(const eris::State &state) {
    EXPECT_EQ(state.options().min_clients(), min_clients);
    EXPECT_EQ(state.options().rounds(), rounds);
    EXPECT_EQ(state.options().split_seed(), split_seed);
    EXPECT_EQ(state.options().splits(), splits);
  }

  void ContainsState(const eris::StateResponse &res) {
    EXPECT_FALSE(res.has_error());
    EXPECT_TRUE(res.has_state());
  }

  void ContainsError(const eris::StateResponse &res, eris::ErrorCode code,
                     const std::string &msg) {
    EXPECT_TRUE(res.has_error());
    EXPECT_FALSE(res.has_state());
    EXPECT_EQ(res.error().code(), code);
    EXPECT_STREQ(res.error().msg().c_str(), msg.c_str());
  }

  inline MockZMQSocket &GetRouter(void) {
    return server_->service_.get_router();
  }
  inline MockZMQSocket &GetPublisher(void) {
    return server_->service_.get_publisher();
  }

  std::shared_ptr<ErisCoordinator<MockZMQSocket>> server_;
  std::thread server_thread_;
  std::future<void> start_ready_;
};

TEST_F(ErisCoordinatorTest, Initialization) {
  std::string router = "tcp://" + address + ":" + std::to_string(router_port);
  std::string publisher =
      "tcp://" + address + ":" + std::to_string(publish_port);
  EXPECT_STREQ(GetRouter().get_endpoint().c_str(), router.c_str());
  EXPECT_STREQ(GetPublisher().get_endpoint().c_str(), publisher.c_str());
}

TEST_F(ErisCoordinatorTest, JoinClient) {
  eris::StateResponse res;
  zmq_msg_t msg;

  TestJoin(eris::JoinRequest{}, &res);

  ContainsState(res);
  const eris::State &state = res.state();
  ValidateTrainingOptions(state);
  EXPECT_FALSE(state.has_assigned_fragment());
  EXPECT_EQ(state.aggregators_size(), 0);

  zmq_msg_init(&msg);
  EXPECT_FALSE(GetPublisher().send_dequeue(&msg, 100));
  zmq_msg_close(&msg);
}

TEST_F(ErisCoordinatorTest, JoinAggregator) {
  const std::string submit_address = "tcp://127.0.0.1:50052";
  const std::string publish_address = "tcp://127.0.0.1:50052";
  zmq_msg_t msg;
  eris::JoinRequest req;
  eris::StateResponse res;
  eris::FragmentInfo info;

  req.set_submit_address(submit_address);
  req.set_publish_address(publish_address);

  TestJoin(req, &res);
  ContainsState(res);
  const eris::State &state = res.state();
  ValidateTrainingOptions(state);
  EXPECT_TRUE(state.has_assigned_fragment());
  EXPECT_EQ(state.aggregators_size(), 1);
  EXPECT_GE(state.assigned_fragment(), 0);
  EXPECT_LT(state.assigned_fragment(), splits);
  bool found = false;

  for (auto aggr : state.aggregators())
    if (aggr.publish_address() == publish_address &&
        aggr.submit_address() == submit_address)
      found = true;

  EXPECT_TRUE(found);

  zmq_msg_init(&msg);
  EXPECT_TRUE(GetPublisher().send_dequeue(&msg));
  EXPECT_TRUE(info.ParseFromArray(zmq_msg_data(&msg), zmq_msg_size(&msg)));
  EXPECT_EQ(info.publish_address(), publish_address);
  EXPECT_EQ(info.submit_address(), submit_address);
  EXPECT_LT(info.id(), splits);
  zmq_msg_close(&msg);
}

TEST_F(ErisCoordinatorTest, JoinTooManyAggregators) {
  std::set<std::pair<std::string, std::string>> aggregators;
  zmq_msg_t msg;
  const uint16_t pull_base = 50052;
  const uint16_t pub_base = 5555;

  for (uint16_t i = 0; i < splits; ++i)
    aggregators.emplace("tcp://127.0.0.1:" + std::to_string(pull_base + i),
                        "tcp://127.0.0.1:" + std::to_string(pub_base + i));

  for (const std::pair<std::string, std::string> &aggregator : aggregators) {
    eris::StateResponse res;
    eris::JoinRequest req;

    req.set_submit_address(aggregator.first);
    req.set_publish_address(aggregator.second);

    TestJoin(req, &res);
    ContainsState(res);
    const eris::State &state = res.state();
    ValidateTrainingOptions(state);
    EXPECT_TRUE(state.has_assigned_fragment());
    EXPECT_GE(state.assigned_fragment(), 0);
    EXPECT_LT(state.assigned_fragment(), splits);

    bool found = false;

    for (auto aggr : state.aggregators())
      if (aggr.publish_address() == aggregator.second &&
          aggr.submit_address() == aggregator.first)
        found = true;

    EXPECT_TRUE(found);
  }

  {
    eris::FragmentInfo info;
    std::set<std::pair<std::string, std::string>> addresses;
    std::unordered_set<uint32_t> ids;

    for (size_t i = 0; i < aggregators.size(); ++i) {
      zmq_msg_init(&msg);
      EXPECT_TRUE(GetPublisher().send_dequeue(&msg));
      EXPECT_TRUE(info.ParseFromArray(zmq_msg_data(&msg), zmq_msg_size(&msg)));
      std::pair<std::string, std::string> aggregator{info.submit_address(),
                                                     info.publish_address()};
      EXPECT_NE(aggregators.find(aggregator), aggregators.end());
      EXPECT_LT(info.id(), splits);
      addresses.insert(aggregator);
      ids.insert(info.id());
      zmq_msg_close(&msg);
    }
    EXPECT_EQ(addresses.size(), aggregators.size());
    EXPECT_EQ(ids.size(), splits);
  }

  {
    eris::JoinRequest req;
    eris::StateResponse res;
    req.set_submit_address("tcp://127.0.0.1:" +
                           std::to_string(pull_base + aggregators.size() + 1));
    req.set_publish_address("tcp://127.0.0.1:" +
                            std::to_string(pub_base + aggregators.size() + 1));

    TestJoin(eris::JoinRequest{}, &res);

    ContainsState(res);
    const eris::State &state = res.state();
    ValidateTrainingOptions(state);
    EXPECT_FALSE(state.has_assigned_fragment());

    for (auto aggr : state.aggregators()) {
      std::pair<std::string, std::string> aggregator{aggr.submit_address(),
                                                     aggr.publish_address()};
      if (aggregators.find(aggregator) == aggregators.end())
        FAIL();
    }
  }

  zmq_msg_init(&msg);
  EXPECT_FALSE(GetPublisher().send_dequeue(&msg, 100));
  zmq_msg_close(&msg);
}

TEST_F(ErisCoordinatorTest, MissingPublishAddress) {
  eris::JoinRequest req;
  eris::StateResponse res;
  zmq_msg_t msg;

  req.set_submit_address("tcp://127.0.0.1:50051");

  TestJoin(req, &res);
  ContainsError(res, eris::ErrorCode::INVALID_ARGUMENT,
                "Missing model updates publishing address");

  zmq_msg_init(&msg);
  EXPECT_FALSE(GetPublisher().send_dequeue(&msg, 100));
  zmq_msg_close(&msg);
}

TEST_F(ErisCoordinatorTest, MissingSubmitAddress) {
  eris::JoinRequest req;
  eris::StateResponse res;
  zmq_msg_t msg;

  req.set_publish_address("tcp://127.0.0.1:50051");

  TestJoin(req, &res);
  ContainsError(res, eris::ErrorCode::INVALID_ARGUMENT,
                "Missing weight submission address");

  zmq_msg_init(&msg);
  EXPECT_FALSE(GetPublisher().send_dequeue(&msg, 100));
  zmq_msg_close(&msg);
}

TEST_F(ErisCoordinatorTest, InvalidPublishAddress) {
  eris::JoinRequest req;
  eris::StateResponse res;
  // zmq_msg_t msg;

  req.set_publish_address("Some random string");
  req.set_submit_address("tcp://127.0.0.1:50051");

  TestJoin(req, &res);
  ContainsError(res, eris::ErrorCode::INVALID_ARGUMENT,
                "A model updates publishing address must have the form "
                "tcp://<address>:<port> where address is a valid IPv4 address");

  // zmq_msg_init(&msg);
  // EXPECT_FALSE(GetPublisher().send_dequeue(&msg, 100));
  // zmq_msg_close(&msg);
}

TEST_F(ErisCoordinatorTest, InvalidSubmitAddress) {
  eris::JoinRequest req;
  eris::StateResponse res;
  // zmq_msg_t msg;

  req.set_publish_address("tcp://127.0.0.1:5000");
  req.set_submit_address("Some random string");

  TestJoin(req, &res);
  ContainsError(
      res, eris::ErrorCode::INVALID_ARGUMENT,
      "A weight submission address must have the form tcp://<address>:<port>"
      "where address is a valid IPv4 address");

  // zmq_msg_init(&msg);
  // EXPECT_FALSE(GetPublisher().send_dequeue(&msg, 100));
  // zmq_msg_close(&msg);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  google::protobuf::ShutdownProtobufLibrary();
  return ret;
}
