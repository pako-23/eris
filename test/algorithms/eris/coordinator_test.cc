#include "algorithms/eris/builder.h"
#include "algorithms/eris/coordinator.h"
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <gtest/gtest.h>
#include <memory>
#include <thread>

class ErisCoordinatorTest : public testing::Test {
protected:
  ErisCoordinatorTest(void) : server{nullptr}, server_thread{nullptr} {
    ErisCoordinatorBuilder builder;
    builder.add_rounds(10);
    builder.add_min_clients(10);
    builder.add_splits(5);

    server.reset(new ErisCoordinator(builder));
    server_thread.reset(
        new std::thread{[](std::shared_ptr<ErisCoordinator> coordinator) {
                          coordinator->start();
                        },
                        server});
  }

  ~ErisCoordinatorTest(void) {
    server->stop();
    server_thread->join();
  }

  std::unique_ptr<std::thread> server_thread;
  std::shared_ptr<ErisCoordinator> server;
};

TEST_F(ErisCoordinatorTest, JoinSingleClient) {}

// TEST_F(ErisCoordinatorTest, JoinAggregator){}
