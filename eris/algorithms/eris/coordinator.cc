#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include "zmq.hpp"
#include <grpcpp/support/status.h>
#include <memory>
#include <string>

ErisCoordinator::ErisCoordinator(const coordinator::TrainingOptions &options,
                                 const std::string &address, uint16_t rpc_port,
                                 uint16_t pub_port)
    : grpc_address_{address + ":" + std::to_string(rpc_port)},
      pub_address_{address + ":" + std::to_string(pub_port)}, options_{options},
      zmq_context_{}, zmq_socket_{zmq_context_, zmq::socket_type::pub} {}

ErisCoordinator::~ErisCoordinator(void) {}

void ErisCoordinator::start(void) {
  CoordinatorImpl service{shared_from_this()};

  ServerBuilder builder;
  builder.AddListeningPort(grpc_address_, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server{builder.BuildAndStart()};

  zmq_socket_.bind(pub_address_);

  server->Wait();
}

ErisCoordinator::CoordinatorImpl::CoordinatorImpl(
    std::shared_ptr<ErisCoordinator> c)
    : coordinator_{c} {}

grpc::ServerUnaryReactor *
ErisCoordinator::CoordinatorImpl::Join(CallbackServerContext *context,
                                       const coordinator::Endpoint *request,
                                       coordinator::JoinResponse *response) {
  if (request->has_publish_port()) {
    for (const coordinator::Endpoint *aggregator : coordinator_->aggregators_)
      if (aggregator && aggregator->address() == request->address()) {
        grpc::ServerUnaryReactor *reactor = context->DefaultReactor();
        reactor->Finish(
            Status(StatusCode::ALREADY_EXISTS,
                   "An aggregator on the provided address already exists"));
        return reactor;
      }
  }

  for (auto i = 0; i < coordinator_->aggregators_.size(); ++i) {
    if (request->has_publish_port() && !coordinator_->aggregators_[i]) {
      response->set_assigned_fragment(i);
      coordinator_->aggregators_[i] = new coordinator::Endpoint{};
      *coordinator_->aggregators_[i] = *request;
      coordinator::FragmentInfo info{};
      info.set_id(i);
      info.set_allocated_address(coordinator_->aggregators_[i]);
      coordinator_->zmq_socket_.send(zmq::buffer(info.SerializeAsString()),
                                     zmq::send_flags::dontwait);
    } else {
      coordinator::FragmentInfo *info = response->add_aggregators();
      info->set_id(i);
      *info->mutable_address() = *coordinator_->aggregators_[i];
    }
  }

  *response->mutable_options() = coordinator_->options_;
  response->set_events_address(coordinator_->pub_address_);

  grpc::ServerUnaryReactor *reactor = context->DefaultReactor();
  reactor->Finish(Status::OK);
  return reactor;
}
