#include "algorithms/eris/client.h"
#include "algorithms/eris/aggregator.grpc.pb.h"
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
#include "spdlog/spdlog.h"
#include "zmq.hpp"
#include <chrono>
#include <condition_variable>
#include <cstddef>
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
    : zmq_context_{}, coordinator_thread_{nullptr}, options_{}, aggregators_{},
      subscriptions_{}, aggregation_mutex_{}, known_aggregators_{0},
      all_aggregators_connected_{}, aggregator_thread_{nullptr},
      aggregator_builder_{aggregator_builder}, splitter_{} {}

void ErisClient::start(const std::string &coordinator_address) {
  std::shared_ptr<grpc::Channel> channel{grpc::CreateChannel(
      coordinator_address, grpc::InsecureChannelCredentials())};

  if (!channel->WaitForConnected(std::chrono::system_clock::now() + timeout)) {
    spdlog::error("Failed to connect to coordinator on {0}",
                  coordinator_address);
    return;
  }

  ClientImpl client{channel, shared_from_this()};
  if (!client.Join())
    return;
  spdlog::info("Successfully joined the training");
  std::unique_lock<std::mutex> lk{aggregation_mutex_};
  all_aggregators_connected_.wait(
      lk, [this] { return known_aggregators_ >= options_.splits(); });

  for (uint32_t round{0}; round < options_.rounds(); ++round) {
    fit();
    submit_parameters(round);
    receive_parameters(round);
    set_parameters(splitter_.get_parameters());
    evaluate();
  }
}

void ErisClient::submit_parameters(uint32_t round) {
  splitter_.split(get_parameters(), round);

  grpc::ClientContext context;
  std::vector<aggregator::Empty> responses(splitter_.size());
  std::mutex mu;
  std::condition_variable cv;
  uint32_t completed{0};

  for (size_t i{0}; i < splitter_.size(); ++i)
    aggregators_[i]->async()->SubmitWeights(
        &context, &splitter_[i], &responses[i],
        [&mu, &cv, &completed](Status status) {
          if (!status.ok())
            spdlog::error("Weight submission failed: {0}",
                          status.error_message());

          std::lock_guard<std::mutex> lock{mu};
          ++completed;
          cv.notify_one();
        });

  std::unique_lock<std::mutex> lock{mu};
  cv.wait(lock, [&completed, this] { return completed == splitter_.size(); });
}

void ErisClient::receive_parameters(uint32_t round) {
  zmq::poller_t<aggregator::WeightUpdate> poller;
  size_t expected{subscriptions_.size()};

  std::vector<aggregator::WeightUpdate> weights(subscriptions_.size());
  std::chrono::minutes timeout{1};
  for (size_t i{0}; i < subscriptions_.size(); ++i)
    poller.add(subscriptions_[i], zmq::event_flags::pollin, &weights[i]);

  std::vector<zmq::poller_event<aggregator::WeightUpdate>> events(
      subscriptions_.size());

  while (expected) {
    const auto n = poller.wait_all(events, timeout);
    if (!n) {
      continue;
    }
    for (size_t i{0}; i < n; ++i) {
      if (events[i].events != zmq::event_flags::pollin)
        continue;

      if (events[i].user_data->round() < round)
        continue;

      if (expected > 0)
        --expected;
      splitter_.reassemble(events[i].user_data, i);
    }
  }
}

bool ErisClient::aggregator_connect(const coordinator::FragmentInfo &info) {
  if (info.id() >= aggregators_.size()) {
    spdlog::error("Received information about unexpected aggregator");
    return false;
  }

  std::shared_ptr<grpc::Channel> channel{grpc::CreateChannel(
      info.submit_address(), grpc::InsecureChannelCredentials())};

  if (!channel->WaitForConnected(std::chrono::system_clock::now() + timeout)) {
    spdlog::error("Failed to connect to aggregator on {0}",
                  info.submit_address());
    return false;
  }

  aggregators_[info.id()] = aggregator::Aggregator::NewStub(channel);
  subscriptions_[info.id()].connect(info.publish_address());

  return true;
}

void ErisClient::start_aggregator(const ErisAggregatorBuilder &builder) {
  std::string grpc_address_{builder.get_rpc_address()};
  AggregatorImpl service{builder};
  ServerBuilder server_builder;
  server_builder.AddListeningPort(grpc_address_,
                                  grpc::InsecureServerCredentials());
  server_builder.RegisterService(&service);

  std::unique_ptr<Server> server{server_builder.BuildAndStart()};
  spdlog::info("Started aggregator gRPC server on {0}", grpc_address_);

  server->Wait();
}

void ErisClient::listen_coordinator_events(const std::string &publish_address) {
  zmq::socket_t sock{zmq_context_, zmq::socket_type::sub};
  sock.connect(publish_address);

  zmq::poller_t<coordinator::FragmentInfo> poller;
  coordinator::FragmentInfo info;

  poller.add(sock, zmq::event_flags::pollin, &info);

  std::vector<zmq::poller_event<coordinator::FragmentInfo>> events(1);
  spdlog::info("Listening on coordinator events on ZeroMQ address {0}",
               publish_address);

  while (true) {
    const auto n = poller.wait_all(events, std::chrono::milliseconds{1000});

    if (!n) {
      std::lock_guard<std::mutex> lk{aggregation_mutex_};
      if (known_aggregators_ >= options_.splits())
        all_aggregators_connected_.notify_one();

      continue;
    }

    if (!aggregator_connect(info)) {
      break;
    }
    {
      std::lock_guard<std::mutex> lk{aggregation_mutex_};
      if (++known_aggregators_ >= options_.splits())
        all_aggregators_connected_.notify_one();
    }
  }
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
          done = true;
          cv.notify_one();
          return;
        }

        this->client_->options_ = response.options();
        this->client_->aggregators_.resize(response.options().splits());
        this->client_->subscriptions_.resize(0);
        for (uint32_t i{0}; i < response.options().splits(); ++i)
          this->client_->subscriptions_.emplace_back(zmq::socket_t{
              this->client_->zmq_context_, zmq::socket_type::sub});

        for (const coordinator::FragmentInfo &aggregator :
             response.aggregators())
          if (!this->client_->aggregator_connect(aggregator)) {
            result = false;
            done = true;
            cv.notify_one();
            return;
          }

        std::lock_guard<std::mutex> lock{mu};
        done = true;
        cv.notify_one();
      });

  std::unique_lock<std::mutex> lock{mu};
  cv.wait(lock, [&done] { return done; });

  if (!result)
    return false;

  this->client_->splitter_.setup(this->client_->get_parameters(),
                                 this->client_->options_.splits(),
                                 this->client_->options_.split_seed());
  this->client_->coordinator_thread_.reset(
      new std::thread{&ErisClient::listen_coordinator_events, this->client_,
                      response.events_address()});

  if (response.has_assigned_fragment()) {
    this->client_->aggregator_builder_.value().add_min_clients(
        response.options().min_clients());
    this->client_->aggregator_builder_.value().add_block_size(
        this->client_->splitter_.get_block_size(response.assigned_fragment()));
    this->client_->aggregator_thread_.reset(
        new std::thread{&ErisClient::start_aggregator, this->client_,
                        this->client_->aggregator_builder_.value()});
  }

  return result;
}

ErisClient::AggregatorImpl::AggregatorImpl(const ErisAggregatorBuilder &builder)
    : current_round_{0}, min_clients_{builder.min_clients_}, weight_update_{},
      zmq_context_{}, zmq_socket_{zmq_context_, zmq::socket_type::pub} {
  zmq_socket_.bind(builder.get_zmq_listen_address());
  spdlog::info("Started aggregator ZeroMQ publisher on {0}",
               builder.get_zmq_listen_address());
  weight_update_.set_contributors(0);
  for (uint32_t i{0}; i < builder.block_size_; ++i)
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
  weight_update_.set_round(++current_round_);

  return reactor;
}
