#include "util/networking.h"
#include <gtest/gtest.h>

TEST(ValidateIPv4, ValidIPv4) {
  EXPECT_TRUE(valid_ipv4("1.1.1.1"));
  EXPECT_TRUE(valid_ipv4("0.0.0.0"));
  EXPECT_TRUE(valid_ipv4("255.255.255.255"));
  EXPECT_TRUE(valid_ipv4("1.0.0.0"));
  EXPECT_TRUE(valid_ipv4("1.255.255.255"));
  EXPECT_TRUE(valid_ipv4("236.255.255.255"));
  EXPECT_TRUE(valid_ipv4("236.236.255.255"));
  EXPECT_TRUE(valid_ipv4("200.255.100.190"));
}

TEST(ValidateIPv4, Strings) {
  EXPECT_FALSE(valid_ipv4("Some text"));
  EXPECT_FALSE(valid_ipv4("Not a valid IPv4 address"));
}

TEST(ValidateIPv4, IPv4InsideString) {
  EXPECT_FALSE(valid_ipv4("1.1.1.1 some text"));
  EXPECT_FALSE(valid_ipv4("Some text 1.1.1.1 some text"));
  EXPECT_FALSE(valid_ipv4("Some text 1.1.1.1"));
  EXPECT_FALSE(valid_ipv4(""));
}

TEST(ValidateIPv4, OutOfRangeIPv4) {
  EXPECT_FALSE(valid_ipv4("1.1.1.256"));
  EXPECT_FALSE(valid_ipv4("1.1.256.1"));
  EXPECT_FALSE(valid_ipv4("1.256.1.1"));
  EXPECT_FALSE(valid_ipv4("256.1.1.1"));
  EXPECT_FALSE(valid_ipv4("256.256.256.256"));
  EXPECT_FALSE(valid_ipv4("300.0.0.0"));
  EXPECT_FALSE(valid_ipv4("0.300.0.0"));
  EXPECT_FALSE(valid_ipv4("0.0.300.0"));
  EXPECT_FALSE(valid_ipv4("0.0.0.300"));
  EXPECT_FALSE(valid_ipv4("300.300.300.300"));
}

TEST(ValidateIPv4, AddressLength) {
  EXPECT_FALSE(valid_ipv4("192.168.1.1.1"));
  EXPECT_FALSE(valid_ipv4("192.168.1"));
  EXPECT_FALSE(valid_ipv4("192.168"));
  EXPECT_FALSE(valid_ipv4("192"));
}

TEST(ValidateIPv4, ZerosBeforeDigits) {
  EXPECT_FALSE(valid_ipv4("01.1.1.1"));
  EXPECT_FALSE(valid_ipv4("1.01.1.1"));
  EXPECT_FALSE(valid_ipv4("1.1.01.1"));
  EXPECT_FALSE(valid_ipv4("1.1.1.01"));
  EXPECT_FALSE(valid_ipv4("01.01.01.01"));
}

TEST(ValidateAggregator, ValidIPv4) {
  EXPECT_TRUE(valid_aggregator("1.1.1.1:100"));
  EXPECT_TRUE(valid_aggregator("0.0.0.0:100"));
  EXPECT_TRUE(valid_aggregator("255.255.255.255:100"));
  EXPECT_TRUE(valid_aggregator("1.0.0.0:100"));
  EXPECT_TRUE(valid_aggregator("1.255.255.255:100"));
  EXPECT_TRUE(valid_aggregator("236.255.255.255:100"));
  EXPECT_TRUE(valid_aggregator("236.236.255.255:100"));
  EXPECT_TRUE(valid_aggregator("200.255.100.190:100"));
}

TEST(ValidateAggregator, Strings) {
  EXPECT_FALSE(valid_ipv4("Some text"));
  EXPECT_FALSE(valid_ipv4("Not a valid aggregator address"));
}

TEST(ValidateAggregator, AggregatorAddressInsideString) {
  EXPECT_FALSE(valid_ipv4("1.1.1.1:100 some text"));
  EXPECT_FALSE(valid_ipv4("Some text 1.1.1.1:100 some text"));
  EXPECT_FALSE(valid_ipv4("Some text 1.1.1.1:100"));
  EXPECT_FALSE(valid_ipv4(""));
}

TEST(ValidateAggregator, OutOfRangeIPv4) {
  EXPECT_FALSE(valid_ipv4("1.1.1.256:100"));
  EXPECT_FALSE(valid_ipv4("1.1.256.1:100"));
  EXPECT_FALSE(valid_ipv4("1.256.1.1:100"));
  EXPECT_FALSE(valid_ipv4("256.1.1.1:100"));
  EXPECT_FALSE(valid_ipv4("256.256.256.256:100"));
}

TEST(ValidateAggregator, AddressLength) {
  EXPECT_FALSE(valid_ipv4("192.168.1.1.1:100"));
  EXPECT_FALSE(valid_ipv4("192.168.1:100"));
  EXPECT_FALSE(valid_ipv4("192.168:100"));
  EXPECT_FALSE(valid_ipv4("192:100"));
}

TEST(ValidateAggregator, ZerosBeforeDigits) {
  EXPECT_FALSE(valid_ipv4("01.1.1.1:100"));
  EXPECT_FALSE(valid_ipv4("1.01.1.1:100"));
  EXPECT_FALSE(valid_ipv4("1.1.01.1:100"));
  EXPECT_FALSE(valid_ipv4("1.1.1.01:100"));
  EXPECT_FALSE(valid_ipv4("01.01.01.01:100"));

  EXPECT_FALSE(valid_ipv4("1.1.1.1:010"));
}

TEST(ValidateAggregator, MissingPortNumber) {
  EXPECT_FALSE(valid_aggregator("1.1.1.1:"));
  EXPECT_FALSE(valid_aggregator("0.0.0.0:"));
  EXPECT_FALSE(valid_aggregator("255.255.255.255:"));
  EXPECT_FALSE(valid_aggregator("1.0.0.0:"));
  EXPECT_FALSE(valid_aggregator("1.255.255.255:"));
  EXPECT_FALSE(valid_aggregator("236.255.255.255:"));
}

TEST(ValidateAggregator, ValidPortNumbers) {
  EXPECT_TRUE(valid_aggregator("1.1.1.1:1"));
  EXPECT_TRUE(valid_aggregator("1.1.1.1:9"));
  EXPECT_TRUE(valid_aggregator("1.1.1.1:90"));
  EXPECT_TRUE(valid_aggregator("1.1.1.1:192"));
  EXPECT_TRUE(valid_aggregator("1.1.1.1:65535"));
  EXPECT_TRUE(valid_aggregator("1.1.1.1:65530"));
  EXPECT_TRUE(valid_aggregator("1.1.1.1:65510"));
  EXPECT_TRUE(valid_aggregator("1.1.1.1:65500"));
  EXPECT_TRUE(valid_aggregator("1.1.1.1:64100"));
  EXPECT_TRUE(valid_aggregator("1.1.1.1:65100"));
  EXPECT_TRUE(valid_aggregator("1.1.1.1:6550"));
  EXPECT_TRUE(valid_aggregator("1.1.1.1:6410"));
  EXPECT_TRUE(valid_aggregator("1.1.1.1:6510"));
  EXPECT_TRUE(valid_aggregator("1.1.1.1:65100"));
}

TEST(ValidateAggregator, InvalidPortNumbers) {
  EXPECT_FALSE(valid_aggregator("1.1.1.1:0"));
  EXPECT_FALSE(valid_aggregator("1.1.1.1:65536"));
  EXPECT_FALSE(valid_aggregator("1.1.1.1:65540"));
  EXPECT_FALSE(valid_aggregator("1.1.1.1:75540"));
  EXPECT_FALSE(valid_aggregator("1.1.1.1:6410000"));
}
