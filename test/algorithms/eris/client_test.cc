#include "algorithms/eris/client_test_helpers.h"
#include <gtest/gtest.h>

class ErisClientTest : public testing::Test {
protected:
  ErisClientTest(void) {}
  ~ErisClientTest(void) {}

  ErisMockClient client;
};

TEST_F(ErisClientTest, JoinMissingRPCAddress) {
  EXPECT_TRUE(client.set_coordinator_subscription("tcp://127.0.0.0:5000"));
  EXPECT_FALSE(client.start());
}

TEST_F(ErisClientTest, JoinMissingPublishAddress) {
  EXPECT_TRUE(client.set_coordinator_rpc("127.0.0.0:5000"));
  EXPECT_FALSE(client.start());
}

TEST_F(ErisClientTest, SetInvalidRPCAddress) {
  EXPECT_TRUE(client.set_coordinator_subscription("tcp://127.0.0.1:1231"));
  EXPECT_FALSE(client.set_coordinator_rpc("invalid address"));
  EXPECT_FALSE(client.start());
}

TEST_F(ErisClientTest, SetInvalidPublishAddress) {
  EXPECT_TRUE(client.set_coordinator_rpc("127.0.0.1:1231"));
  EXPECT_FALSE(client.set_coordinator_subscription("invalid address"));
  EXPECT_FALSE(client.start());
}

TEST_F(ErisClientTest, SetValidAggregatorConfig) {
  EXPECT_TRUE(client.set_aggregator_config("127.0.0.1", 1231, 120));
}

TEST_F(ErisClientTest, SetInvalidAggregatorConfig) {
  EXPECT_FALSE(client.set_aggregator_config("127.0.1", 1231, 1323));
  EXPECT_FALSE(client.set_aggregator_config("0.0.0.0", 1231, 1323));
  EXPECT_FALSE(client.set_aggregator_config("127.0.0.1", 1231, 0));
  EXPECT_FALSE(client.set_aggregator_config("127.0.0.1", 0, 1323));
  EXPECT_FALSE(client.set_aggregator_config("127.0.0.1", 1231, 1231));
}
