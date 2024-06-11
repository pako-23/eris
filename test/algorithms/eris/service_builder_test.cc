#include "algorithms/eris/builder.h"
#include <gtest/gtest.h>

class ErisServiceBuilderTest : public testing::Test {
protected:
  ErisServiceBuilderTest(void) {}

  ErisServiceBuilder builder;
};

TEST_F(ErisServiceBuilderTest, DefaultBuilder) {
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
}

TEST_F(ErisServiceBuilderTest, ConfigureAddress) {
  EXPECT_TRUE(builder.add_listen_address("192.168.0.1"));
  EXPECT_EQ(builder.get_rpc_listen_address(), "192.168.0.1:5051");
}

TEST_F(ErisServiceBuilderTest, InvalidIPAddress) {
  EXPECT_FALSE(builder.add_listen_address("not an Ipaddress"));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
}

TEST_F(ErisServiceBuilderTest, ConfigurePort) {
  EXPECT_TRUE(builder.add_rpc_port(8080));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:8080");
}

TEST_F(ErisServiceBuilderTest, InvalidPortNumber) {
  EXPECT_FALSE(builder.add_rpc_port(0));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
}

TEST_F(ErisServiceBuilderTest, ConfigurePortAndAddress) {
  EXPECT_TRUE(builder.add_listen_address("192.168.0.1"));
  EXPECT_TRUE(builder.add_rpc_port(8080));
  EXPECT_EQ(builder.get_rpc_listen_address(), "192.168.0.1:8080");
}
