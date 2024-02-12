#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>

#include "../coordinator.h"
#include "coordinator.grpc.pb.h"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;

class ErisCoordinator final : public Coordinator {
 public:
  ErisCoordinator(void);
  ~ErisCoordinator(void);
  void run(void) override;

 private:
  class JoinRequest {
   public:
    JoinRequest(coordinator::Coordinator::AsyncService*,
                ServerCompletionQueue*);
    void proceed(void);

   private:
    coordinator::Coordinator::AsyncService* service_;
    ServerCompletionQueue* queue_;
    ServerContext ctx_;

    coordinator::JoinRequest request_;
    coordinator::JoinResponse reply_;
    ServerAsyncResponseWriter<coordinator::JoinResponse> responder_;

    enum Status { CREATE, PROCESS, FINISH };
    Status status_;
  };

  std::unique_ptr<Server> server_;
  coordinator::Coordinator::AsyncService service_;
  std::unique_ptr<ServerCompletionQueue> queue_;
};
