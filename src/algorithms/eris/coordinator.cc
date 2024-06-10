#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include "spdlog/spdlog.h"
// #include "zmq.hpp"
#include <grpcpp/support/status.h>
#include <memory>
#include <string>

ErisCoordinator::ErisCoordinator(const ErisCoordinatorBuilder &builder)
    : builder_{builder} {}

ErisCoordinator::~ErisCoordinator(void) {}

void ErisCoordinator::start(void) {
  CoordinatorImpl service{builder_.get_options()};

  std::string grpc_address{builder_.get_rpc_listen_address()};

  ServerBuilder builder;
  builder.AddListeningPort(grpc_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server{builder.BuildAndStart()};
  spdlog::info("started gRPC server on {0}", grpc_address);

  server->Wait();
}

ErisCoordinator::ClientConnection::ClientConnection(std::shared_ptr<Channel>) {}

ErisCoordinator::CoordinatorImpl::CoordinatorImpl(
    const coordinator::TrainingOptions &options)
    : options_{options}, clients_{}, aggregators_(options.splits()) {}

grpc::ServerUnaryReactor *
ErisCoordinator::CoordinatorImpl::Join(CallbackServerContext *context,
                                       const coordinator::JoinRequest *request,
                                       coordinator::JoinResponse *response) {
  grpc::ServerUnaryReactor *reactor{context->DefaultReactor()};

  // TODO: try to connect to the client

  bool needs_fragment{request->can_aggregate()};

  for (std::vector<std::string>::size_type i = 0; i < aggregators_.size();
       ++i) {
    if (needs_fragment && aggregators_[i].empty()) {
      response->set_assigned_fragment(i);
      aggregators_[i] = request->register_address();
      needs_fragment = false;

      spdlog::info("registered new aggregator for fragment {} at {} ", i,
                   request->register_address());
      // TODO: inform all clients about the new aggregator

    } else if (!aggregators_[i].empty()) {
      // coordinator::FragmentInfo *info = response->add_aggregators();
      // info->set_id(i);
      // info->set_aggregator(aggregators_[i]);
    }
  }

  *response->mutable_options() = options_;
  reactor->Finish(Status::OK);

  return reactor;
}
