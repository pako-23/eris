#include "util/networking.h"
#include <gtest/gtest.h>

TEST(ValidateAggregatorPublish, ValidIPv4) {
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:100"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://255.255.255.255:100"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.0.0.0:100"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.255.255.255:100"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://236.255.255.255:100"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://236.236.255.255:100"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://200.255.100.190:100"));
}

TEST(ValidateAggregatorPublish, Strings) {
  EXPECT_FALSE(valid_aggregator_publish("Some text"));
  EXPECT_FALSE(valid_aggregator_publish("Not a valid aggregator address"));
}

TEST(ValidateAggregatorPublish, AggregatorAddressInsideString) {
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.1.1.1:100 some text"));
  EXPECT_FALSE(
      valid_aggregator_publish("Some text tcp://1.1.1.1:100 some text"));
  EXPECT_FALSE(valid_aggregator_publish("Some text tcp://1.1.1.1:100"));
  EXPECT_FALSE(valid_aggregator_publish(""));
}

TEST(ValidateAggregatorPublish, OutOfRangeIPv4) {
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.1.1.256:100"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.1.256.1:100"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.256.1.1:100"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://256.1.1.1:100"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://256.256.256.256:100"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://0.0.0.0:100"));
}

TEST(ValidateAggregatorPublish, AddressLength) {
  EXPECT_FALSE(valid_aggregator_publish("tcp://192.168.1.1.1:100"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://192.168.1:100"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://192.168:100"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://192:100"));
}

TEST(ValidateAggregatorPublish, ZerosBeforeDigits) {
  EXPECT_FALSE(valid_aggregator_publish("tcp://01.1.1.1:100"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.01.1.1:100"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.1.01.1:100"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.1.1.01:100"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://01.01.01.01:100"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.1.1.1:010"));
}

TEST(ValidateAggregatorPublish, MissingProto) {
  EXPECT_FALSE(valid_aggregator_publish("1.1.1.1:100"));
  EXPECT_FALSE(valid_aggregator_publish("255.255.255.255:100"));
  EXPECT_FALSE(valid_aggregator_publish("1.0.0.0:100"));
  EXPECT_FALSE(valid_aggregator_publish("1.255.255.255:100"));
  EXPECT_FALSE(valid_aggregator_publish("236.255.255.255:100"));
  EXPECT_FALSE(valid_aggregator_publish("236.236.255.255:100"));
  EXPECT_FALSE(valid_aggregator_publish("200.255.100.190:100"));
}

TEST(ValidateAggregatorPublish, MissingPortNumber) {
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.1.1.1:"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://0.0.0.0:"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://255.255.255.255:"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.0.0.0:"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.255.255.255:"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://236.255.255.255:"));
}

TEST(ValidateAggregatorPublish, ValidPortNumbers) {
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:1"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:9"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:90"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:192"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:65535"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:65530"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:65510"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:65500"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:64100"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:65100"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:6550"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:6410"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:6510"));
  EXPECT_TRUE(valid_aggregator_publish("tcp://1.1.1.1:65100"));
}

TEST(ValidateAggregatorPublish, InvalidPortNumbers) {
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.1.1.1:0"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.1.1.1:65536"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.1.1.1:65540"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.1.1.1:75540"));
  EXPECT_FALSE(valid_aggregator_publish("tcp://1.1.1.1:6410000"));
}
