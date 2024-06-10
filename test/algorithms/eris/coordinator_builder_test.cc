#include "algorithms/eris/builder.h"
#include <gtest/gtest.h>

TEST(ErisCoordinatorBuilderTest, default_configuration) {
  ErisCoordinatorBuilder builder;

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST(ErisCoordinatorBuilderTest, change_address) {
  ErisCoordinatorBuilder builder;

  EXPECT_TRUE(builder.add_listen_address("192.168.0.1"));

  EXPECT_EQ(builder.get_rpc_listen_address(), "192.168.0.1:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST(ErisCoordinatorBuilderTest, invalid_ip_address) {
  ErisCoordinatorBuilder builder;

  EXPECT_FALSE(builder.add_listen_address("not an Ipaddress"));

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST(ErisCoordinatorBuilderTest, change_port) {
  ErisCoordinatorBuilder builder;

  EXPECT_TRUE(builder.add_rpc_port(8080));

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:8080");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST(ErisCoordinatorBuilderTest, invalid_port_number) {
  ErisCoordinatorBuilder builder;

  EXPECT_FALSE(builder.add_rpc_port(0));

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST(ErisCoordinatorBuilderTest, change_port_and_address) {
  ErisCoordinatorBuilder builder;

  EXPECT_TRUE(builder.add_listen_address("192.168.0.1"));
  EXPECT_TRUE(builder.add_rpc_port(8080));

  EXPECT_EQ(builder.get_rpc_listen_address(), "192.168.0.1:8080");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST(ErisCoordinatorBuilderTest, change_rounds) {
  ErisCoordinatorBuilder builder;

  EXPECT_TRUE(builder.add_rounds(10));

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 10);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST(ErisCoordinatorBuilderTest, invalid_rounds) {
  ErisCoordinatorBuilder builder;

  EXPECT_FALSE(builder.add_rounds(0));

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST(ErisCoordinatorBuilderTest, change_splits) {
  ErisCoordinatorBuilder builder;

  EXPECT_TRUE(builder.add_splits(15));

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 15);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST(ErisCoordinatorBuilderTest, invalid_splits) {
  ErisCoordinatorBuilder builder;

  EXPECT_FALSE(builder.add_splits(0));

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST(ErisCoordinatorBuilderTest, change_min_clients) {
  ErisCoordinatorBuilder builder;

  EXPECT_TRUE(builder.add_min_clients(5));

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 5);
}

TEST(ErisCoordinatorBuilderTest, invalid_min_clients) {
  ErisCoordinatorBuilder builder;

  EXPECT_FALSE(builder.add_min_clients(0));

  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}
