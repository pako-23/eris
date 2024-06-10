#include "algorithms/eris/builder.h"
#include <gtest/gtest.h>

TEST(ErisServiceBuilderTest, default_configuration) {
  ErisServiceBuilder builder;

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
}

TEST(ErisServiceBuilderTest, change_address) {
  ErisServiceBuilder builder;

  EXPECT_TRUE(builder.add_listen_address("192.168.0.1"));

  EXPECT_EQ(builder.get_rpc_listen_address(), "192.168.0.1:5051");
}

TEST(ErisServiceBuilderTest, invalid_ip_address) {
  ErisServiceBuilder builder;

  EXPECT_FALSE(builder.add_listen_address("not an Ipaddress"));

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
}

TEST(ErisServiceBuilderTest, change_port) {
  ErisServiceBuilder builder;

  EXPECT_TRUE(builder.add_rpc_port(8080));

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:8080");
}

TEST(ErisServiceBuilderTest, invalid_port_number) {
  ErisServiceBuilder builder;

  EXPECT_FALSE(builder.add_rpc_port(0));

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
}

TEST(ErisServiceBuilderTest, change_port_and_address) {
  ErisServiceBuilder builder;

  EXPECT_TRUE(builder.add_listen_address("192.168.0.1"));
  EXPECT_TRUE(builder.add_rpc_port(8080));

  EXPECT_EQ(builder.get_rpc_listen_address(), "192.168.0.1:8080");
}
