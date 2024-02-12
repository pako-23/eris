#include "coordinator.h"

ErisCoordinator::ErisCoordinator(void) {}

ErisCoordinator::~ErisCoordinator(void) {
  if (server_) server_->Shutdown();
  if (queue_) queue_->Shutdown();
}

void ErisCoordinator::run(void) {
  std::string address = "0.0.0.0:50051";

  ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service_);

  queue_ = builder.AddCompletionQueue();
  server_ = builder.BuildAndStart();

  new JoinRequest(&service_, queue_.get());
  void* tag;
  bool ok{true};

  while (ok) {
    if (!queue_->Next(&tag, &ok)) break;

    static_cast<JoinRequest*>(tag)->proceed();
  }
}

ErisCoordinator::JoinRequest::JoinRequest(
    coordinator::Coordinator::AsyncService* service,
    ServerCompletionQueue* queue)
    : service_{service}, queue_{queue}, responder_(&ctx_), status_(CREATE) {
  proceed();
}

void ErisCoordinator::JoinRequest::proceed(void) {
  switch (status_) {
    case CREATE:
      status_ = PROCESS;
      service_->RequestJoin(&ctx_, &request_, &responder_, queue_, queue_,
                            this);
      break;
    case PROCESS:
      new JoinRequest(service_, queue_);

      reply_.set_model("model");

      status_ = FINISH;
      responder_.Finish(reply_, grpc::Status::OK, this);
      break;
    default:
      delete this;
      break;
  }
}
