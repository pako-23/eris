#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include "spdlog/spdlog.h"
#include "util/util.h"
#include "zmq.hpp"
#include <grpcpp/support/status.h>
#include <memory>
#include <string>

ErisCoordinator::ErisCoordinator(const coordinator::TrainingOptions &options,
                                 const std::string &address, uint16_t rpc_port,
                                 uint16_t pub_port)
    : grpc_address_{address + ":" + std::to_string(rpc_port)},
      zmq_listening_address_{zmq_listen_address(address, pub_port)},
      zmq_publish_address_{zmq_publish_address(address, pub_port)},
      options_{options}, zmq_context_{},
      zmq_socket_{zmq_context_, zmq::socket_type::pub} {}

ErisCoordinator::~ErisCoordinator(void) {}

void ErisCoordinator::start(void) {
  CoordinatorImpl service{shared_from_this()};

  zmq_socket_.bind(zmq_publish_address_);
  spdlog::info("Started ZeroMQ publisher on {0}", zmq_publish_address_);

  ServerBuilder builder;
  builder.AddListeningPort(grpc_address_, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server{builder.BuildAndStart()};
  spdlog::info("Started gRPC server on {0}", grpc_address_);

  server->Wait();
}

ErisCoordinator::CoordinatorImpl::CoordinatorImpl(
    std::shared_ptr<ErisCoordinator> c)
    : coordinator_{c} {}

grpc::ServerUnaryReactor *
ErisCoordinator::CoordinatorImpl::Join(CallbackServerContext *context,
                                       const coordinator::JoinRequest *request,
                                       coordinator::JoinResponse *response) {
  if (request->has_aggregator()) {
    for (const auto &aggregator : coordinator_->aggregators_)
      if (aggregator && aggregator->id == request->aggregator().address()) {
        grpc::ServerUnaryReactor *reactor = context->DefaultReactor();
        reactor->Finish(
            Status(StatusCode::ALREADY_EXISTS,
                   "An aggregator on the provided address already exists"));
        return reactor;
      }
  }

  for (auto i = 0; i < coordinator_->aggregators_.size(); ++i) {
    if (request->has_aggregator() && !coordinator_->aggregators_[i]) {
      response->set_assigned_fragment(i);
      coordinator_->aggregators_[i] =
          std::make_unique<Aggregator>(request->aggregator());

      coordinator::FragmentInfo info{};
      info.set_id(i);
      info.set_publish_address(coordinator_->aggregators_[i]->publish_address);
      info.set_publish_address(coordinator_->aggregators_[i]->submit_address);
      coordinator_->zmq_socket_.send(zmq::buffer(info.SerializeAsString()),
                                     zmq::send_flags::dontwait);
    } else {
      coordinator::FragmentInfo *info = response->add_aggregators();
      info->set_id(i);
      info->set_publish_address(coordinator_->aggregators_[i]->publish_address);
      info->set_publish_address(coordinator_->aggregators_[i]->submit_address);
    }
  }

  *response->mutable_options() = coordinator_->options_;
  response->set_events_address(coordinator_->zmq_publish_address_);
  grpc::ServerUnaryReactor *reactor = context->DefaultReactor();
  reactor->Finish(Status::OK);
  return reactor;
}

ErisCoordinator::Aggregator::Aggregator(const coordinator::Aggregator &aggr)
    : publish_address{"tcp://" + aggr.address() + ":" +
                      std::to_string(aggr.submit_port())},
      submit_address{aggr.address() + ":" + std::to_string(aggr.submit_port())},
      id{aggr.address()} {}
