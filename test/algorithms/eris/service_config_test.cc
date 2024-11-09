#include "algorithms/eris/config.h"
#include <gtest/gtest.h>

class ErisServiceConfigTest : public testing::Test {
protected:
  ErisServiceConfigTest(void) : config{} {}

  ErisServiceConfig config;
};

TEST_F(ErisServiceConfigTest, DefaultConfiguration) {
  EXPECT_EQ(config.get_router().get_endpoint(), "tcp://*:0");
  EXPECT_EQ(config.get_publisher().get_endpoint(), "tcp://*:0");
}

TEST_F(ErisServiceConfigTest, ChangeAddress) {
  EXPECT_TRUE(config.set_router_address("192.168.0.1"));
  EXPECT_TRUE(config.set_publish_address("192.168.0.1"));
  EXPECT_EQ(config.get_router().get_endpoint(), "tcp://192.168.0.1:0");
  EXPECT_EQ(config.get_publisher().get_endpoint(), "tcp://192.168.0.1:0");
}

TEST_F(ErisServiceConfigTest, InvalidIPAddress) {
  EXPECT_FALSE(config.set_router_address("not an Ipaddress"));
  EXPECT_FALSE(config.set_publish_address("not an Ipaddress"));
  EXPECT_EQ(config.get_router().get_endpoint(), "tcp://*:0");
  EXPECT_EQ(config.get_publisher().get_endpoint(), "tcp://*:0");
}

TEST_F(ErisServiceConfigTest, ChangePort) {
  config.set_router_port(8080);
  config.set_publish_port(8081);
  EXPECT_EQ(config.get_router().get_endpoint(), "tcp://*:8080");
  EXPECT_EQ(config.get_publisher().get_endpoint(), "tcp://*:8081");
}

TEST_F(ErisServiceConfigTest, ConfigurePortAndAddress) {
  EXPECT_TRUE(config.set_router_address("192.168.0.1"));
  config.set_router_port(8080);
  config.set_publish_port(8081);
  EXPECT_TRUE(config.set_publish_address("192.168.0.1"));
  EXPECT_EQ(config.get_router().get_endpoint(), "tcp://192.168.0.1:8080");
  EXPECT_EQ(config.get_publisher().get_endpoint(), "tcp://192.168.0.1:8081");
}
