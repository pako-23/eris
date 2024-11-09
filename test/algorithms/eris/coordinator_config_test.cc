#include "algorithms/eris/config.h"
#include <gtest/gtest.h>

class ErisCoordinatorConfigTest : public testing::Test {
protected:
  ErisCoordinatorConfigTest(void) {}

  ErisCoordinatorConfig config;
};

TEST_F(ErisCoordinatorConfigTest, DefaultConfiguration) {
  EXPECT_EQ(config.get_router().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_ROUTER_PORT));
  EXPECT_EQ(config.get_publisher().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBLISH_PORT));
  EXPECT_EQ(config.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(config.get_options().splits(), 1);
  EXPECT_EQ(config.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorConfigTest, ChangeAddress) {
  EXPECT_TRUE(config.set_router_address("192.168.0.1"));
  EXPECT_TRUE(config.set_publish_address("192.168.0.1"));
  EXPECT_EQ(config.get_router().get_endpoint(),
            "tcp://192.168.0.1:" + std::to_string(DEFAULT_ERIS_ROUTER_PORT));
  EXPECT_EQ(config.get_publisher().get_endpoint(),
            "tcp://192.168.0.1:" + std::to_string(DEFAULT_ERIS_PUBLISH_PORT));
  EXPECT_EQ(config.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(config.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(config.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorConfigTest, InvalidIPAddress) {
  EXPECT_FALSE(config.set_router_address("not an Ipaddress"));
  EXPECT_FALSE(config.set_publish_address("not an Ipaddress"));
  EXPECT_EQ(config.get_router().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_ROUTER_PORT));
  EXPECT_EQ(config.get_publisher().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBLISH_PORT));
  EXPECT_EQ(config.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(config.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(config.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorConfigTest, ChangePort) {
  config.set_router_port(8080);
  config.set_publish_port(8081);
  EXPECT_EQ(config.get_router().get_endpoint(), "tcp://*:8080");
  EXPECT_EQ(config.get_publisher().get_endpoint(), "tcp://*:8081");
  EXPECT_EQ(config.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(config.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(config.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorConfigTest, ConfigurePortAndAddress) {
  EXPECT_TRUE(config.set_router_address("192.168.0.1"));
  config.set_router_port(8080);
  config.set_publish_port(8081);
  EXPECT_TRUE(config.set_publish_address("192.168.0.1"));
  EXPECT_EQ(config.get_router().get_endpoint(), "tcp://192.168.0.1:8080");
  EXPECT_EQ(config.get_publisher().get_endpoint(), "tcp://192.168.0.1:8081");
  EXPECT_EQ(config.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(config.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(config.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorConfigTest, ConfigureRounds) {
  EXPECT_TRUE(config.set_rounds(10));
  EXPECT_EQ(config.get_router().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_ROUTER_PORT));
  EXPECT_EQ(config.get_options().rounds(), 10);
  EXPECT_EQ(config.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(config.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorConfigTest, InvalidRounds) {
  EXPECT_FALSE(config.set_rounds(0));
  EXPECT_EQ(config.get_router().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_ROUTER_PORT));
  EXPECT_EQ(config.get_publisher().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBLISH_PORT));
  EXPECT_EQ(config.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(config.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(config.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorConfigTest, ConfigureSplits) {
  EXPECT_TRUE(config.set_splits(15));
  EXPECT_EQ(config.get_router().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_ROUTER_PORT));
  EXPECT_EQ(config.get_publisher().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBLISH_PORT));
  EXPECT_EQ(config.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(config.get_options().splits(), 15);
  EXPECT_EQ(config.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorConfigTest, InvalidSplits) {
  EXPECT_FALSE(config.set_splits(0));
  EXPECT_EQ(config.get_router().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_ROUTER_PORT));
  EXPECT_EQ(config.get_publisher().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBLISH_PORT));
  EXPECT_EQ(config.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(config.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(config.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorConfigTest, ConfigureMinClients) {
  EXPECT_TRUE(config.set_min_clients(5));
  EXPECT_EQ(config.get_router().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_ROUTER_PORT));
  EXPECT_EQ(config.get_publisher().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBLISH_PORT));
  EXPECT_EQ(config.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(config.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(config.get_options().min_clients(), 5);
}

TEST_F(ErisCoordinatorConfigTest, InvalidMinClients) {
  EXPECT_FALSE(config.set_min_clients(0));
  EXPECT_EQ(config.get_router().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_ROUTER_PORT));
  EXPECT_EQ(config.get_publisher().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBLISH_PORT));
  EXPECT_EQ(config.get_options().rounds(), 1);
  EXPECT_EQ(config.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(config.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
}

TEST_F(ErisCoordinatorConfigTest, ConfigureSplitSeed) {
  config.set_split_seed(10);
  EXPECT_EQ(config.get_router().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_ROUTER_PORT));
  EXPECT_EQ(config.get_publisher().get_endpoint(),
            "tcp://*:" + std::to_string(DEFAULT_ERIS_PUBLISH_PORT));
  EXPECT_EQ(config.get_options().rounds(), DEFAULT_ERIS_ROUNDS);
  EXPECT_EQ(config.get_options().splits(), DEFAULT_ERIS_SPLITS);
  EXPECT_EQ(config.get_options().min_clients(), DEFAULT_ERIS_MIN_CLIENTS);
  EXPECT_EQ(config.get_options().split_seed(), 10);
}

TEST_F(ErisCoordinatorConfigTest, DefaultSplitSeed) {
  ErisCoordinatorConfig second_config;

  EXPECT_EQ(config.get_router().get_endpoint(),
            second_config.get_router().get_endpoint());
  EXPECT_EQ(config.get_options().rounds(),
            second_config.get_options().rounds());
  EXPECT_EQ(config.get_options().splits(),
            second_config.get_options().splits());
  EXPECT_EQ(config.get_options().min_clients(),
            second_config.get_options().min_clients());
  EXPECT_NE(config.get_options().split_seed(),
            second_config.get_options().split_seed());
}
