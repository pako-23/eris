#include "mock_client.h"
#include <gtest/gtest.h>
#include <stdexcept>

TEST(ErisClientInitialize, EmptyCoordinatorRouter) {
  try {
    MockClient client{"", "tcp://127.0.0.0:5000"};

    FAIL() << "expected invalid_argument exception";
  } catch (const std::invalid_argument &e) {
    EXPECT_EQ(
        e.what(),
        std::string{"invalid endpoint address for coordinator router socket"});
  } catch (...) {
    FAIL() << "expected invalid_argument exception";
  }
}

TEST(ErisClientInitialize, EmptyCoordinatorPublish) {
  try {
    MockClient client{"tcp://127.0.0.1:5000", ""};

    FAIL() << "expected invalid_argument exception";
  } catch (const std::invalid_argument &e) {
    EXPECT_EQ(
        e.what(),
        std::string{"invalid endpoint address for coordinator publish socket"});
  } catch (...) {
    FAIL() << "expected invalid_argument exception";
  }
}

TEST(ErisClientInitialize, InvalidCoordinatorRouter) {
  try {
    MockClient client{"invalid address", "tcp://127.0.0.0:5000"};

    FAIL() << "expected invalid_argument exception";
  } catch (const std::invalid_argument &e) {
    EXPECT_EQ(
        e.what(),
        std::string{"invalid endpoint address for coordinator router socket"});
  } catch (...) {
    FAIL() << "expected invalid_argument exception";
  }
}

TEST(ErisClientInitialize, InvalidCoordinatorPublish) {
  try {
    MockClient client{"tcp://127.0.0.0:5000", "invalid address"};

    FAIL() << "expected invalid_argument exception";
  } catch (const std::invalid_argument &e) {
    EXPECT_EQ(
        e.what(),
        std::string{"invalid endpoint address for coordinator publish socket"});
  } catch (...) {
    FAIL() << "expected invalid_argument exception";
  }
}

TEST(ErisClientInitialize, ValidAggregatorConfig) {
  MockClient client{};

  EXPECT_TRUE(client.get_subscriber().subscribed());
  EXPECT_FALSE(client.get_dealer().subscribed());
  EXPECT_TRUE(client.set_aggregator_config("127.0.0.1", 1231, 120));

  EXPECT_EQ(client.get_aggr_address(), "127.0.0.1");
  EXPECT_EQ(client.get_aggr_submit_port(), 1231);
  EXPECT_EQ(client.get_aggr_publish_port(), 120);
}

TEST(ErisClientInitialize, AggregatorZeroPublishPort) {
  MockClient client{};

  EXPECT_TRUE(client.set_aggregator_config("127.0.0.1", 1231, 0));
  EXPECT_EQ(client.get_aggr_address(), "127.0.0.1");
  EXPECT_EQ(client.get_aggr_submit_port(), 1231);
  EXPECT_EQ(client.get_aggr_publish_port(), 0);
}

TEST(ErisClientInitialize, AggregatorZeroSubmitPort) {
  MockClient client{};

  EXPECT_TRUE(client.set_aggregator_config("127.0.0.1", 0, 1323));
  EXPECT_EQ(client.get_aggr_address(), "127.0.0.1");
  EXPECT_EQ(client.get_aggr_submit_port(), 0);
  EXPECT_EQ(client.get_aggr_publish_port(), 1323);
}

TEST(ErisClientInitialize, AggregatorZeroPorts) {
  MockClient client{};

  EXPECT_TRUE(client.set_aggregator_config("127.0.0.1", 0, 0));
  EXPECT_EQ(client.get_aggr_address(), "127.0.0.1");
  EXPECT_EQ(client.get_aggr_submit_port(), 0);
  EXPECT_EQ(client.get_aggr_publish_port(), 0);
}

TEST(ErisClientInitialize, AggregatorDefaultPorts) {
  MockClient client{};

  EXPECT_TRUE(client.set_aggregator_config("127.0.0.1"));
  EXPECT_EQ(client.get_aggr_address(), "127.0.0.1");
  EXPECT_EQ(client.get_aggr_submit_port(), 0);
  EXPECT_EQ(client.get_aggr_publish_port(), 0);
}

TEST(ErisClientInitialize, AggregatorDefaultPublishPort) {
  MockClient client{};

  EXPECT_TRUE(client.set_aggregator_config("127.0.0.1", 20));
  EXPECT_EQ(client.get_aggr_address(), "127.0.0.1");
  EXPECT_EQ(client.get_aggr_submit_port(), 20);
  EXPECT_EQ(client.get_aggr_publish_port(), 0);
}

TEST(ErisClientInitialize, InvalidAggregatorConfig) {
  MockClient client{};

  EXPECT_TRUE(client.get_subscriber().subscribed());
  EXPECT_FALSE(client.get_dealer().subscribed());
  EXPECT_FALSE(client.set_aggregator_config("127.0.1", 1231, 1323));
  EXPECT_EQ(client.get_aggr_address(), "");
  EXPECT_EQ(client.get_aggr_submit_port(), 0);
  EXPECT_EQ(client.get_aggr_publish_port(), 0);
  EXPECT_FALSE(client.set_aggregator_config("0.0.0.0", 1231, 1323));
  EXPECT_EQ(client.get_aggr_address(), "");
  EXPECT_EQ(client.get_aggr_submit_port(), 0);
  EXPECT_EQ(client.get_aggr_publish_port(), 0);

  EXPECT_FALSE(client.set_aggregator_config("127.0.0.1", 1231, 1231));
  EXPECT_EQ(client.get_aggr_address(), "");
  EXPECT_EQ(client.get_aggr_submit_port(), 0);
  EXPECT_EQ(client.get_aggr_publish_port(), 0);
}
