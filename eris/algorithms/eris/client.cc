#include "algorithms/eris/client.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "spdlog/spdlog.h"
#include "zmq.hpp"
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <grpcpp/support/status.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

using grpc::Status;

static std::chrono::minutes timeout = std::chrono::minutes(1);

ErisClient::ErisClient(const std::string &coordinator_address,
                       std::optional<AggregatorConfig> aggregator_opts)
    : coordinator_addr_{coordinator_address}, zmq_context_{},
      publisher_sock_{zmq_context_, zmq::socket_type::sub}, subscriptions_{},
      aggregator_{nullptr}, aggregator_config_{aggregator_opts} {
  // TODO: Validate parameters
}

void ErisClient::start(void) {
  std::shared_ptr<grpc::Channel> channel{grpc::CreateChannel(
      coordinator_addr_, grpc::InsecureChannelCredentials())};

  if (!channel->WaitForConnected(std::chrono::system_clock::now() + timeout)) {
    spdlog::error("Failed to connect the coordinator on {0}",
                  coordinator_addr_);
    return;
  }

  ClientImpl client{channel, shared_from_this()};

  if (!client.Join())
    return;
  spdlog::info("Successfully joined the training");

  fit();

  get_parameters();
}

bool ErisClient::start_aggregator(void) { return true; }

ErisClient::ClientImpl::ClientImpl(std::shared_ptr<Channel> channel,
                                   std::shared_ptr<ErisClient> client)
    : stub_{coordinator::Coordinator::NewStub(channel)}, client_{client} {}

bool ErisClient::ClientImpl::Join(void) {
  coordinator::JoinRequest request;
  if (this->client_->aggregator_config_.has_value()) {
    request.mutable_aggregator()->set_address(
        this->client_->aggregator_config_.value().address);
    request.mutable_aggregator()->set_submit_port(
        this->client_->aggregator_config_.value().submit_port);
    request.mutable_aggregator()->set_publish_port(
        this->client_->aggregator_config_.value().publish_port);
  }

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
          spdlog::error("Failed to join the training: {0}",
                        status.error_message());
          result = false;
          return;
        }
        this->client_->options_ = response.options();
        this->client_->aggregators_.resize(response.options().splits());
        this->client_->subscriptions_.resize(0);
        for (uint32_t i{0}; i < response.options().splits(); ++i)
          this->client_->subscriptions_.emplace_back(zmq::socket_t{
              this->client_->zmq_context_, zmq::socket_type::sub});
        this->client_->publisher_sock_.connect(response.events_address());

        for (const auto &aggregator : response.aggregators()) {
          if (aggregator.id() < 0 &&
              aggregator.id() >= response.options().splits()) {

            result = false;
            return;
          }
          this->client_->aggregators_[aggregator.id()] =
              aggregator.submit_address();
          this->client_->subscriptions_[aggregator.id()].connect(
              aggregator.publish_address());
        }

        if (response.has_assigned_fragment() &&
            !this->client_->start_aggregator())
          result = false;
      });
  std::unique_lock<std::mutex> lock(m);
  c.wait(lock, [&done] { return done; });
  return result;
}
