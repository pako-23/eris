#include "algorithms/eris/builder.h"
#include <gtest/gtest.h>

class ErisCoordinatorBuilderTest : public testing::Test {
protected:
  ErisCoordinatorBuilderTest(void) {}

  ErisCoordinatorBuilder builder;
};

TEST_F(ErisCoordinatorBuilderTest, DefaultConfiguration) {
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST_F(ErisCoordinatorBuilderTest, ChangeAddress) {
  EXPECT_TRUE(builder.add_listen_address("192.168.0.1"));
  EXPECT_EQ(builder.get_rpc_listen_address(), "192.168.0.1:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST_F(ErisCoordinatorBuilderTest, InvalidIPAddress) {
  EXPECT_FALSE(builder.add_listen_address("not an Ipaddress"));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST_F(ErisCoordinatorBuilderTest, ChangePort) {
  EXPECT_TRUE(builder.add_rpc_port(8080));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:8080");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST_F(ErisCoordinatorBuilderTest, InvalidPortNumber) {
  EXPECT_FALSE(builder.add_rpc_port(0));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST_F(ErisCoordinatorBuilderTest, ConfigurePortAndAddress) {
  EXPECT_TRUE(builder.add_listen_address("192.168.0.1"));
  EXPECT_TRUE(builder.add_rpc_port(8080));
  EXPECT_EQ(builder.get_rpc_listen_address(), "192.168.0.1:8080");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST_F(ErisCoordinatorBuilderTest, ConfigureRounds) {
  EXPECT_TRUE(builder.add_rounds(10));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 10);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST_F(ErisCoordinatorBuilderTest, InvalidRounds) {
  EXPECT_FALSE(builder.add_rounds(0));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST_F(ErisCoordinatorBuilderTest, ConfigureSplits) {
  EXPECT_TRUE(builder.add_splits(15));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 15);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST_F(ErisCoordinatorBuilderTest, InvalidSplits) {
  EXPECT_FALSE(builder.add_splits(0));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}

TEST_F(ErisCoordinatorBuilderTest, ConfigureMinClients) {
  EXPECT_TRUE(builder.add_min_clients(5));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 5);
}

TEST_F(ErisCoordinatorBuilderTest, InvalidMinClients) {
  EXPECT_FALSE(builder.add_min_clients(0));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:5051");
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), 3);
}
