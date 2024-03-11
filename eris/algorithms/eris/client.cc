#include "algorithms/eris/client.h"
#include "algorithms/eris/aggregator.pb.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/support/server_callback.h"
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pytypes.h"
#include "spdlog/spdlog.h"
#include "zmq.hpp"
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <grpc/grpc.h>
#include <grpcpp/support/status.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unistd.h>

using grpc::Status;

static std::chrono::minutes timeout = std::chrono::minutes(1);

ErisClient::ErisClient(std::optional<ErisAggregatorBuilder> aggregator_builder)
    : zmq_context_{}, publisher_sock_{zmq_context_, zmq::socket_type::sub},
      options_{}, aggregators_{}, subscriptions_{}, aggregator_{nullptr},
      aggregator_builder_{aggregator_builder} {}

void ErisClient::start(const std::string &coordinator_address) {
  std::shared_ptr<grpc::Channel> channel{grpc::CreateChannel(
      coordinator_address, grpc::InsecureChannelCredentials())};

  if (!channel->WaitForConnected(std::chrono::system_clock::now() + timeout)) {
    spdlog::error("Failed to connect the coordinator on {0}",
                  coordinator_address);
    return;
  }

  ClientImpl client{channel, shared_from_this()};

  if (!client.Join())
    return;

  spdlog::info("Successfully joined the training");

  fit();
  for (py::handle handle : get_parameters()) {
    py::array_t<double> array{py::cast<py::array>(handle).request()};
    spdlog::info("Dimension -> {0}", array.ndim());
  }
  while (true) {
    sleep(1000);
  }
}

void ErisClient::start_aggregator(const ErisAggregatorBuilder &builder) {
  std::string grpc_address_{builder.get_rpc_address()};
  AggregatorImpl service{builder};
  ServerBuilder server_builder;
  server_builder.AddListeningPort(grpc_address_,
                                  grpc::InsecureServerCredentials());
  server_builder.RegisterService(&service);

  std::unique_ptr<Server> server{server_builder.BuildAndStart()};
  spdlog::info("Started aggrgator gRPC server on {0}", grpc_address_);

  server->Wait();
}

ErisClient::ClientImpl::ClientImpl(std::shared_ptr<Channel> channel,
                                   std::shared_ptr<ErisClient> client)
    : stub_{coordinator::Coordinator::NewStub(channel)}, client_{client} {}

bool ErisClient::ClientImpl::Join(void) {
  coordinator::JoinRequest request;
  if (this->client_->aggregator_builder_.has_value()) {
    request.mutable_aggregator()->set_address(
        this->client_->aggregator_builder_.value().listen_address_);
    request.mutable_aggregator()->set_submit_port(
        this->client_->aggregator_builder_.value().rpc_port_);
    request.mutable_aggregator()->set_publish_port(
        this->client_->aggregator_builder_.value().publish_port_);
  }

  coordinator::JoinResponse response;
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + timeout);

  std::mutex mu;
  std::condition_variable cv;
  bool done{false};
  bool result{true};

  stub_->async()->Join(
      &context, &request, &response,
      [&cv, &mu, &done, &result, &response, this](Status status) {
        if (!status.ok()) {
          spdlog::error("Failed to join the training: {0}",
                        status.error_message());
          result = false;
          std::lock_guard<std::mutex> lock{mu};
          done = true;
          cv.notify_all();
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
            std::lock_guard<std::mutex> lock{mu};
            done = true;
            cv.notify_all();
            return;
          }
          this->client_->aggregators_[aggregator.id()] =
              aggregator.submit_address();
          this->client_->subscriptions_[aggregator.id()].connect(
              aggregator.publish_address());
        }

        if (response.has_assigned_fragment()) {

          this->client_->aggregator_builder_.value().add_min_clients(
              response.options().min_clients());
          this->client_->aggregator_.reset(
              new std::thread{&ErisClient::start_aggregator, this->client_,
                              this->client_->aggregator_builder_.value()});
        }

        std::lock_guard<std::mutex> lock{mu};
        done = true;
        cv.notify_all();
      });

  std::unique_lock<std::mutex> lock{mu};
  while (!done)
    cv.wait(lock, [&done] { return done; });

  return result;
}

ErisClient::AggregatorImpl::AggregatorImpl(const ErisAggregatorBuilder &builder)
    : current_round_{0}, min_clients_{builder.min_clients_}, weight_update_{},
      zmq_context_{}, zmq_socket_{zmq_context_, zmq::socket_type::pub} {
  zmq_socket_.bind(builder.get_zmq_listen_address());

  weight_update_.set_contributors(0);
  for (int i{0}; i < 10; ++i)
    weight_update_.add_weight(0.0);
}

grpc::ServerUnaryReactor *ErisClient::AggregatorImpl::SubmitWeights(
    CallbackServerContext *context, const aggregator::Weight *request,
    [[maybe_unused]] aggregator::Empty *response) {
  grpc::ServerUnaryReactor *reactor{context->DefaultReactor()};

  if (request->round() != 0) {
    reactor->Finish(Status(StatusCode::INVALID_ARGUMENT,
                           "Provided a weight from a wrong round"));
    return reactor;
  } else if (request->weight_size() != weight_update_.weight_size()) {
    reactor->Finish(
        Status(StatusCode::INVALID_ARGUMENT, "Wrong parameter length"));
    return reactor;
  }

  for (int i{0}; i < request->weight_size(); ++i)
    weight_update_.set_weight(i, weight_update_.weight()[i] +
                                     request->weight()[i]);

  weight_update_.set_contributors(weight_update_.contributors() + 1);
  reactor->Finish(Status::OK);

  if (weight_update_.contributors() < min_clients_)
    return reactor;

  zmq_socket_.send(zmq::buffer(weight_update_.SerializePartialAsString()),
                   zmq::send_flags::dontwait);

  for (int i{0}; i < weight_update_.weight_size(); ++i)
    weight_update_.set_weight(i, 0.0);
  weight_update_.set_contributors(0);

  return reactor;
}
