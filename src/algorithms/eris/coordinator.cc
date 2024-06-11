#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include "spdlog/spdlog.h"
#include <grpcpp/support/server_callback.h>
#include <grpcpp/support/status.h>
#include <memory>
#include <string>

ErisCoordinator::ErisCoordinator(const ErisCoordinatorBuilder &builder)
    : builder_{builder}, server_{nullptr} {}

ErisCoordinator::~ErisCoordinator(void) { stop(); }

void ErisCoordinator::start(void) {
  CoordinatorImpl service{builder_.get_options()};
  std::string grpc_address{builder_.get_rpc_listen_address()};

  ServerBuilder builder;

  builder.AddListeningPort(grpc_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  server_.reset(builder.BuildAndStart().get());
  spdlog::info("started gRPC server on {0}", grpc_address);

  server_->Wait();
}

void ErisCoordinator::stop(void) {
  if (server_)
    server_->Shutdown();
}

ErisCoordinator::CoordinatorImpl::CoordinatorImpl(
    const coordinator::TrainingOptions &opt)
    : options_{opt}, aggregators_(opt.splits()) {}

grpc::ServerWriteReactor<coordinator::CoordinatorUpdate> *
ErisCoordinator::CoordinatorImpl::Join(CallbackServerContext *ctx,
                                       const coordinator::JoinRequest *req) {
  return nullptr;
}
