#include "algorithms/eris/client.h"

#include <grpcpp/support/status.h>

#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "zmq.hpp"
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

using grpc::Status;

static std::chrono::minutes timeout = std::chrono::minutes(1);

ErisClient::ErisClient(const std::string &coordinator_address,
                       const std::string &address, uint16_t grpc_port,
                       std::optional<uint16_t> pub_port)
    : bind_addr_{address}, coordinator_addr_{coordinator_address},
      grpc_port_{grpc_port}, pub_port_{pub_port}, publisher_ctx_{},
      publisher_sock_{publisher_ctx_, zmq::socket_type::sub} {
  // TODO: Validate parameters
}

void ErisClient::start(void) {
  std::shared_ptr<grpc::Channel> channel{grpc::CreateChannel(
      coordinator_addr_, grpc::InsecureChannelCredentials())};

  if (!channel->WaitForConnected(std::chrono::system_clock::now() + timeout)) {
    std::cerr << "Failed to connect to the coordinator" << std::endl;
    return;
  }

  ClientImpl client{channel, shared_from_this()};

  if (!client.Join()) {
    std::cerr << "Failed to join the training" << std::endl;
    return;
  }

  std::cout << "Successfully joined the training" << std::endl;
}

ErisClient::ClientImpl::ClientImpl(std::shared_ptr<Channel> channel,
                                   std::shared_ptr<ErisClient> client)
    : stub_{coordinator::Coordinator::NewStub(channel)}, client_{client} {}

bool ErisClient::ClientImpl::Join(void) {
  coordinator::Endpoint request;
  request.set_rpc_port(client_->grpc_port_);
  request.set_address(client_->bind_addr_);
  if (client_->pub_port_.has_value())
    request.set_publish_port(client_->pub_port_.value());

  coordinator::JoinResponse response;
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + timeout);

  std::mutex m;
  std::condition_variable c;
  bool done{false};
  bool result{true};

  stub_->async()->Join(
      &context, &request, &response,
      [&done, &result, &response, this](Status status) {
        done = true;
        if (!status.ok()) {
          std::cerr << status.error_message() << std::endl;
          result = false;
        } else {
          this->client_->options_ = response.options();
          this->client_->aggregators_.resize(response.options().splits());
          this->client_->publisher_sock_.connect(response.events_address());
          if (response.has_assigned_fragment()) {
            // TODO: start aggregator service
          }
          for (const auto &aggregator : response.aggregators()) {
            this->client_->aggregators_[aggregator.id()] = aggregator.address();
            // TODO: connect to zmq address
          }
        }
      });
  std::unique_lock<std::mutex> lock(m);
  c.wait(lock, [&done] { return done; });
  return result;
}
