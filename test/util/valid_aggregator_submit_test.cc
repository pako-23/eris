#include "util/networking.h"
#include <gtest/gtest.h>

TEST(ValidateAggregatorSubmit, ValidIPv4) {
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:100"));
  EXPECT_TRUE(valid_aggregator_submit("255.255.255.255:100"));
  EXPECT_TRUE(valid_aggregator_submit("1.0.0.0:100"));
  EXPECT_TRUE(valid_aggregator_submit("1.255.255.255:100"));
  EXPECT_TRUE(valid_aggregator_submit("236.255.255.255:100"));
  EXPECT_TRUE(valid_aggregator_submit("236.236.255.255:100"));
  EXPECT_TRUE(valid_aggregator_submit("200.255.100.190:100"));
}

TEST(ValidateAggregatorSubmit, Strings) {
  EXPECT_FALSE(valid_ipv4("Some text"));
  EXPECT_FALSE(valid_ipv4("Not a valid aggregator address"));
}

TEST(ValidateAggregatorSubmit, AggregatorAddressInsideString) {
  EXPECT_FALSE(valid_ipv4("1.1.1.1:100 some text"));
  EXPECT_FALSE(valid_ipv4("Some text 1.1.1.1:100 some text"));
  EXPECT_FALSE(valid_ipv4("Some text 1.1.1.1:100"));
  EXPECT_FALSE(valid_ipv4(""));
}

TEST(ValidateAggregatorSubmit, OutOfRangeIPv4) {
  EXPECT_FALSE(valid_ipv4("1.1.1.256:100"));
  EXPECT_FALSE(valid_ipv4("1.1.256.1:100"));
  EXPECT_FALSE(valid_ipv4("1.256.1.1:100"));
  EXPECT_FALSE(valid_ipv4("256.1.1.1:100"));
  EXPECT_FALSE(valid_ipv4("256.256.256.256:100"));
  EXPECT_FALSE(valid_aggregator_submit("0.0.0.0:100"));
}

TEST(ValidateAggregatorSubmit, AddressLength) {
  EXPECT_FALSE(valid_ipv4("192.168.1.1.1:100"));
  EXPECT_FALSE(valid_ipv4("192.168.1:100"));
  EXPECT_FALSE(valid_ipv4("192.168:100"));
  EXPECT_FALSE(valid_ipv4("192:100"));
}

TEST(ValidateAggregatorSubmit, ZerosBeforeDigits) {
  EXPECT_FALSE(valid_ipv4("01.1.1.1:100"));
  EXPECT_FALSE(valid_ipv4("1.01.1.1:100"));
  EXPECT_FALSE(valid_ipv4("1.1.01.1:100"));
  EXPECT_FALSE(valid_ipv4("1.1.1.01:100"));
  EXPECT_FALSE(valid_ipv4("01.01.01.01:100"));

  EXPECT_FALSE(valid_ipv4("1.1.1.1:010"));
}

TEST(ValidateAggregatorSubmit, MissingPortNumber) {
  EXPECT_FALSE(valid_aggregator_submit("1.1.1.1:"));
  EXPECT_FALSE(valid_aggregator_submit("0.0.0.0:"));
  EXPECT_FALSE(valid_aggregator_submit("255.255.255.255:"));
  EXPECT_FALSE(valid_aggregator_submit("1.0.0.0:"));
  EXPECT_FALSE(valid_aggregator_submit("1.255.255.255:"));
  EXPECT_FALSE(valid_aggregator_submit("236.255.255.255:"));
}

TEST(ValidateAggregatorSubmit, ValidPortNumbers) {
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:1"));
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:9"));
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:90"));
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:192"));
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:65535"));
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:65530"));
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:65510"));
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:65500"));
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:64100"));
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:65100"));
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:6550"));
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:6410"));
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:6510"));
  EXPECT_TRUE(valid_aggregator_submit("1.1.1.1:65100"));
}

TEST(ValidateAggregatorSubmit, InvalidPortNumbers) {
  EXPECT_FALSE(valid_aggregator_submit("1.1.1.1:0"));
  EXPECT_FALSE(valid_aggregator_submit("1.1.1.1:65536"));
  EXPECT_FALSE(valid_aggregator_submit("1.1.1.1:65540"));
  EXPECT_FALSE(valid_aggregator_submit("1.1.1.1:75540"));
  EXPECT_FALSE(valid_aggregator_submit("1.1.1.1:6410000"));
}
