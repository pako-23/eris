#include "algorithms/eris/builder.h"
#include <gtest/gtest.h>

class ErisCoordinatorBuilderTest : public testing::Test {
protected:
  ErisCoordinatorBuilderTest(void) {}

  ErisCoordinatorBuilder builder;
};

TEST_F(ErisCoordinatorBuilderTest, DefaultConfiguration) {
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "0.0.0.0:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));
  EXPECT_EQ(builder.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(builder.get_options().splits(), 1);
  EXPECT_EQ(builder.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorBuilderTest, ChangeAddress) {
  EXPECT_TRUE(builder.add_rpc_listen_address("192.168.0.1"));
  EXPECT_TRUE(builder.add_publish_address("192.168.0.1"));
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "192.168.0.1:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://192.168.0.1:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));
  EXPECT_EQ(builder.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(builder.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(builder.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorBuilderTest, InvalidIPAddress) {
  EXPECT_FALSE(builder.add_rpc_listen_address("not an Ipaddress"));
  EXPECT_FALSE(builder.add_publish_address("not an Ipaddress"));
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "0.0.0.0:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));
  EXPECT_EQ(builder.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(builder.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(builder.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorBuilderTest, ChangePort) {
  EXPECT_TRUE(builder.add_rpc_port(8080));
  EXPECT_TRUE(builder.add_publish_port(8081));
  EXPECT_EQ(builder.get_rpc_listen_address(), "0.0.0.0:8080");
  EXPECT_EQ(builder.get_pubsub_listen_address(), "tcp://*:8081");
  EXPECT_EQ(builder.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(builder.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(builder.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorBuilderTest, ConfigurePortAndAddress) {
  EXPECT_TRUE(builder.add_rpc_listen_address("192.168.0.1"));
  EXPECT_TRUE(builder.add_rpc_port(8080));
  EXPECT_TRUE(builder.add_publish_port(8081));
  EXPECT_TRUE(builder.add_publish_address("192.168.0.1"));
  EXPECT_EQ(builder.get_rpc_listen_address(), "192.168.0.1:8080");
  EXPECT_EQ(builder.get_pubsub_listen_address(), "tcp://192.168.0.1:8081");
  EXPECT_EQ(builder.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(builder.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(builder.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorBuilderTest, ConfigureRounds) {
  EXPECT_TRUE(builder.add_rounds(10));
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "0.0.0.0:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_options().rounds(), 10);
  EXPECT_EQ(builder.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(builder.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorBuilderTest, InvalidRounds) {
  EXPECT_FALSE(builder.add_rounds(0));
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "0.0.0.0:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));
  EXPECT_EQ(builder.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(builder.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(builder.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorBuilderTest, ConfigureSplits) {
  EXPECT_TRUE(builder.add_splits(15));
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "0.0.0.0:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));
  EXPECT_EQ(builder.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(builder.get_options().splits(), 15);
  EXPECT_EQ(builder.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorBuilderTest, InvalidSplits) {
  EXPECT_FALSE(builder.add_splits(0));
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "0.0.0.0:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));
  EXPECT_EQ(builder.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(builder.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(builder.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorBuilderTest, ConfigureMinClients) {
  EXPECT_TRUE(builder.add_min_clients(5));
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "0.0.0.0:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));
  EXPECT_EQ(builder.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(builder.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(builder.get_options().min_clients(), 5);
}

TEST_F(ErisCoordinatorBuilderTest, InvalidMinClients) {
  EXPECT_FALSE(builder.add_min_clients(0));
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "0.0.0.0:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));
  EXPECT_EQ(builder.get_options().rounds(), 1);
  EXPECT_EQ(builder.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(builder.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorBuilderTest, ConfigureSplitSeed) {
  EXPECT_TRUE(builder.add_split_seed(10));
  EXPECT_EQ(builder.get_rpc_listen_address(),
            "0.0.0.0:" + std::to_string(DEFAULT_ERIS_RPC_PORT));
  EXPECT_EQ(builder.get_pubsub_listen_address(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBSUB_PORT));
  EXPECT_EQ(builder.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(builder.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(builder.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
  EXPECT_EQ(builder.get_options().split_seed(), 10);
}

TEST_F(ErisCoordinatorBuilderTest, DefaultSplitSeed) {
  ErisCoordinatorBuilder second_builder;

  EXPECT_EQ(builder.get_rpc_listen_address(),
            second_builder.get_rpc_listen_address());
  EXPECT_EQ(builder.get_options().rounds(),
            second_builder.get_options().rounds());
  EXPECT_EQ(builder.get_options().splits(),
            second_builder.get_options().splits());
  EXPECT_EQ(builder.get_options().min_clients(),
            second_builder.get_options().min_clients());
  EXPECT_NE(builder.get_options().split_seed(),
            second_builder.get_options().split_seed());
}
