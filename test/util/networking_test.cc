#include "util/networking.h"
#include <gtest/gtest.h>

TEST(NetworkingTest, ValidIPv4) {
  EXPECT_TRUE(valid_ipv4("1.1.1.1"));
  EXPECT_TRUE(valid_ipv4("0.0.0.0"));
  EXPECT_TRUE(valid_ipv4("255.255.255.255"));
  EXPECT_TRUE(valid_ipv4("1.0.0.0"));
  EXPECT_TRUE(valid_ipv4("1.255.255.255"));
}
