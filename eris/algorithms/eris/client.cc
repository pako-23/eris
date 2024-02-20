#include "algorithms/eris/client.h"

#include <grpcpp/support/status.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

#include "algorithms/eris/coordinator.grpc.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"

using grpc::Status;

static std::chrono::minutes timeout = std::chrono::minutes(1);

ErisClient::ErisClient(const std::string& coordinator_address,
                       const std::string& address, uint16_t port)
    : bind_addr_{address},
      coordinator_addr_{coordinator_address},
      bind_port_{port} {
  // TODO: Validate parameters
}

void ErisClient::run(void) {
  std::shared_ptr<grpc::Channel> channel{grpc::CreateChannel(
      coordinator_addr_, grpc::InsecureChannelCredentials())};

  if (!channel->WaitForConnected(std::chrono::system_clock::now() + timeout)) {
    std::cerr << "Failed to connect to the coordinator" << std::endl;
    return;
  }

  ClientImpl client{channel};

  client.Join(bind_addr_, bind_port_);

  std::cout << "Successfully joined the training" << std::endl;
}

ErisClient::ClientImpl::ClientImpl(std::shared_ptr<Channel> channel)
    : stub_{coordinator::Coordinator::NewStub(channel)} {}

bool ErisClient::ClientImpl::Join(const std::string& addr, uint16_t port) {
  coordinator::JoinRequest request;
  request.set_port(port);
  request.set_address(addr);

  coordinator::JoinResponse response;

  grpc::ClientContext context;

  context.set_deadline(std::chrono::system_clock::now() + timeout);

  std::mutex m;
  std::condition_variable c;
  bool done{false};
  bool result{true};

  stub_->async()->Join(&context, &request, &response,
                       [&done, &result, &response, this](Status status) {
                         done = true;
                         if (!status.ok()) {
                           std::cout << status.error_message() << std::endl;
                           result = false;
                         } else {
                           std::cout << "Joined the training with mode: "
                                     << response.model() << std::endl;
                         }
                       });
  std::unique_lock<std::mutex> lock(m);
  c.wait(lock, [&done] { return done; });
  return result;
}
