#include "algorithms/eris/builder.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>

class ErisAggregatorBuilderTest : public testing::Test {
protected:
  ErisAggregatorBuilderTest(void)
      : dev{}, rng{dev()}, dist{1, std::numeric_limits<uint32_t>::max()},
        expected_fragment_id(dist(rng)), expected_fragment_size(dist(rng)),
        builder(expected_fragment_id, expected_fragment_size) {}

private:
  std::random_device dev;
  std::mt19937 rng;
  std::uniform_int_distribution<std::mt19937::result_type> dist;

protected:
  uint32_t expected_fragment_id;
  size_t expected_fragment_size;
  ErisAggregatorBuilder builder;
};

TEST_F(ErisAggregatorBuilderTest, DefaultConfiguration) {
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "0.0.0.0:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));
  EXPECT_EQ(builder.get_min_client(), DEFAULT_ERIS_MIN_CLIENTS);
  EXPECT_EQ(builder.get_fragment_id(), expected_fragment_id);
  EXPECT_EQ(builder.get_fragment_size(), expected_fragment_size);
}

TEST_F(ErisAggregatorBuilderTest, ChangeAddress) {
  EXPECT_TRUE(builder.add_rpc_listen_address("192.168.0.1"));
  EXPECT_TRUE(builder.add_publish_address("192.168.0.1"));
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "192.168.0.1:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://192.168.0.1:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));

  EXPECT_EQ(builder.get_min_client(), DEFAULT_ERIS_MIN_CLIENTS);
  EXPECT_EQ(builder.get_fragment_id(), expected_fragment_id);
  EXPECT_EQ(builder.get_fragment_size(), expected_fragment_size);
}

TEST_F(ErisAggregatorBuilderTest, InvalidIPAddress) {
  EXPECT_FALSE(builder.add_rpc_listen_address("not an Ipaddress"));
  EXPECT_FALSE(builder.add_publish_address("not an Ipaddress"));
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "0.0.0.0:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));
  EXPECT_EQ(builder.get_min_client(), DEFAULT_ERIS_MIN_CLIENTS);
  EXPECT_EQ(builder.get_fragment_id(), expected_fragment_id);
  EXPECT_EQ(builder.get_fragment_size(), expected_fragment_size);
}

TEST_F(ErisAggregatorBuilderTest, ChangePort) {
  EXPECT_TRUE(builder.add_rpc_port(8080));
  EXPECT_TRUE(builder.add_publish_port(8081));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:8080");
  EXPECT_EQ(builder.get_pubsub_listen_address(), "tcp://*:8081");
  EXPECT_EQ(builder.get_min_client(), DEFAULT_ERIS_MIN_CLIENTS);
  EXPECT_EQ(builder.get_fragment_id(), expected_fragment_id);
  EXPECT_EQ(builder.get_fragment_size(), expected_fragment_size);
}

TEST_F(ErisAggregatorBuilderTest, ConfigurePortAndAddress) {
  EXPECT_TRUE(builder.add_rpc_listen_address("192.168.0.1"));
  EXPECT_TRUE(builder.add_rpc_port(8080));
  EXPECT_TRUE(builder.add_publish_port(8081));
  EXPECT_TRUE(builder.add_publish_address("192.168.0.1"));
  EXPECT_EQ(builder.get_rpc_listen_address(), "192.168.0.1:8080");
  EXPECT_EQ(builder.get_pubsub_listen_address(), "tcp://192.168.0.1:8081");
  EXPECT_EQ(builder.get_min_client(), DEFAULT_ERIS_MIN_CLIENTS);
  EXPECT_EQ(builder.get_fragment_id(), expected_fragment_id);
  EXPECT_EQ(builder.get_fragment_size(), expected_fragment_size);
}

TEST_F(ErisAggregatorBuilderTest, ConfigureMinClients) {
  EXPECT_TRUE(builder.add_min_clients(5));
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "0.0.0.0:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));
  EXPECT_EQ(builder.get_min_client(), 5);
  EXPECT_EQ(builder.get_fragment_id(), expected_fragment_id);
  EXPECT_EQ(builder.get_fragment_size(), expected_fragment_size);
}

TEST_F(ErisAggregatorBuilderTest, InvalidMinClients) {
  EXPECT_FALSE(builder.add_min_clients(0));
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "0.0.0.0:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));
  EXPECT_EQ(builder.get_min_client(), DEFAULT_ERIS_MIN_CLIENTS);
  EXPECT_EQ(builder.get_fragment_id(), expected_fragment_id);
  EXPECT_EQ(builder.get_fragment_size(), expected_fragment_size);
}

TEST_F(ErisAggregatorBuilderTest, InvalidFragmentSize) {
  EXPECT_THROW(ErisAggregatorBuilder builder(1, 0), std::invalid_argument);
}
