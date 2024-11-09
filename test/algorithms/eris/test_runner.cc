#include "google/protobuf/message_lite.h"
#include <gtest/gtest.h>

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();

  google::protobuf::ShutdownProtobufLibrary();
  return ret;
}
