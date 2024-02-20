#include "algorithms/eris/coordinator.h"

#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"

ErisCoordinator::ErisCoordinator(void) {}

void ErisCoordinator::run(void) {
  std::string bind_addr{"0.0.0.0:50051"};

  CoordinatorImpl service;

  ServerBuilder builder;
  builder.AddListeningPort(bind_addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server{builder.BuildAndStart()};

  server->Wait();
}

ErisCoordinator::CoordinatorImpl::CoordinatorImpl(void) {}

grpc::ServerUnaryReactor* ErisCoordinator::CoordinatorImpl::Join(
    CallbackServerContext* context, const coordinator::JoinRequest* request,
    coordinator::JoinResponse* response) {
  std::cout << "Request received" << std::endl;
  response->set_model("model ok");
  grpc::ServerUnaryReactor* reactor = context->DefaultReactor();
  reactor->Finish(Status::OK);
  return reactor;
}
