#include "util/networking.h"
#include <gtest/gtest.h>

TEST(ValidateZMQEndpoint, ValidIPv4) {
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:100"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://255.255.255.255:100"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.0.0.0:100"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.255.255.255:100"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://236.255.255.255:100"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://236.236.255.255:100"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://200.255.100.190:100"));
}

TEST(ValidateZMQEndpoint, Strings) {
  EXPECT_FALSE(valid_zmq_endpoint("Some text"));
  EXPECT_FALSE(valid_zmq_endpoint("Not a valid aggregator address"));
}

TEST(ValidateZMQEndpoint, AddressInsideString) {
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.1.1.1:100 some text"));
  EXPECT_FALSE(valid_zmq_endpoint("Some text tcp://1.1.1.1:100 some text"));
  EXPECT_FALSE(valid_zmq_endpoint("Some text tcp://1.1.1.1:100"));
  EXPECT_FALSE(valid_zmq_endpoint(""));
}

TEST(ValidateZMQEndpoint, OutOfRangeIPv4) {
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.1.1.256:100"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.1.256.1:100"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.256.1.1:100"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://256.1.1.1:100"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://256.256.256.256:100"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://0.0.0.0:100"));
}

TEST(ValidateZMQEndpoint, AddressLength) {
  EXPECT_FALSE(valid_zmq_endpoint("tcp://192.168.1.1.1:100"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://192.168.1:100"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://192.168:100"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://192:100"));
}

TEST(ValidateZMQEndpoint, ZerosBeforeDigits) {
  EXPECT_FALSE(valid_zmq_endpoint("tcp://01.1.1.1:100"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.01.1.1:100"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.1.01.1:100"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.1.1.01:100"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://01.01.01.01:100"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.1.1.1:010"));
}

TEST(ValidateZMQEndpoint, MissingProto) {
  EXPECT_FALSE(valid_zmq_endpoint("1.1.1.1:100"));
  EXPECT_FALSE(valid_zmq_endpoint("255.255.255.255:100"));
  EXPECT_FALSE(valid_zmq_endpoint("1.0.0.0:100"));
  EXPECT_FALSE(valid_zmq_endpoint("1.255.255.255:100"));
  EXPECT_FALSE(valid_zmq_endpoint("236.255.255.255:100"));
  EXPECT_FALSE(valid_zmq_endpoint("236.236.255.255:100"));
  EXPECT_FALSE(valid_zmq_endpoint("200.255.100.190:100"));
}

TEST(ValidateZMQEndpoint, MissingPortNumber) {
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.1.1.1:"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://0.0.0.0:"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://255.255.255.255:"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.0.0.0:"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.255.255.255:"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://236.255.255.255:"));
}

TEST(ValidateZMQEndpoint, ValidPortNumbers) {
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:1"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:9"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:90"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:192"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:65535"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:65530"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:65510"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:65500"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:64100"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:65100"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:6550"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:6410"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:6510"));
  EXPECT_TRUE(valid_zmq_endpoint("tcp://1.1.1.1:65100"));
}

TEST(ValidateZMQEndpoint, InvalidPortNumbers) {
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.1.1.1:0"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.1.1.1:65536"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.1.1.1:65540"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.1.1.1:75540"));
  EXPECT_FALSE(valid_zmq_endpoint("tcp://1.1.1.1:6410000"));
}
