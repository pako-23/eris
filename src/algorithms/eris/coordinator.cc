#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include "spdlog/spdlog.h"
#include "util/networking.h"
#include "zmq.h"
#include <cstddef>
#include <cstring>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/support/server_callback.h>
#include <grpcpp/support/status.h>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

ErisCoordinator::ErisCoordinator(const ErisCoordinatorBuilder &builder)
    : server_{nullptr}, service_{builder.get_options(), this}, started_{false},
      listening_address_{builder.get_rpc_listen_address()} {
  grpc::ServerBuilder srv_builder;
  int port;

  char endpoint[255];
  size_t endpointlen = sizeof(endpoint);

  zmq_ctx = zmq_ctx_new();
  if (!zmq_ctx)
    throw std::bad_alloc();

  publisher = zmq_socket(zmq_ctx, ZMQ_PUB);
  if (!publisher)
    throw std::bad_alloc();

  zmq_bind(publisher, builder.get_pubsub_listen_address().c_str());
  zmq_getsockopt(publisher, ZMQ_LAST_ENDPOINT, &endpoint, &endpointlen);

  pubsub_port_ = atoi(strchr(strchr(endpoint, ':') + 1, ':') + 1);

  srv_builder.AddListeningPort(builder.get_rpc_listen_address(),
                               grpc::InsecureServerCredentials(), &port);
  srv_builder.RegisterService(&service_);

  server_ = srv_builder.BuildAndStart();
  rpc_port_ = port;
}

ErisCoordinator::~ErisCoordinator(void) {
  if (started_)
    stop();
  zmq_close(publisher);
  zmq_ctx_destroy(zmq_ctx);
}

void ErisCoordinator::start(void) {
  started_ = true;
  char endpoint[255];
  size_t endpointlen = sizeof(endpoint);

  zmq_getsockopt(publisher, ZMQ_LAST_ENDPOINT, &endpoint, &endpointlen);

  spdlog::info("listening RPC requests on {0}:{1} and publishing on {2}",
               listening_address_.substr(0, listening_address_.find(':')),
               rpc_port_, endpoint);
  server_->Wait();
}

void ErisCoordinator::stop(void) {
  server_->Shutdown();
  started_ = false;
}

bool ErisCoordinator::publish_aggregator(const FragmentInfo &info) {
  zmq_msg_t msg;

  zmq_msg_init_size(&msg, info.ByteSizeLong());
  info.SerializeToArray(zmq_msg_data(&msg), info.ByteSizeLong());
  bool ret = zmq_msg_send(&msg, publisher, 0) > 0;
  zmq_msg_close(&msg);

  return ret;
}

ErisCoordinator::CoordinatorImpl::CoordinatorImpl(
    const TrainingOptions &options, ErisCoordinator *coordinator)
    : options_{options}, aggregators_(options.splits()), mu_{},
      coordinator_{coordinator} {}

grpc::ServerUnaryReactor *ErisCoordinator::CoordinatorImpl::Join(
    CallbackServerContext *ctx, const JoinRequest *req, InitialState *res) {

  class Reactor : public grpc::ServerUnaryReactor {
  public:
    explicit Reactor(CoordinatorImpl *ctx, const JoinRequest *req,
                     InitialState *res) {
      if (req->has_aggr_address() && !valid_aggregator(req->aggr_address())) {
        Finish(
            Status(StatusCode::INVALID_ARGUMENT,
                   "An aggregator address must have the form <address>:<port> "
                   "where address is a valid IPv4 address"));
        return;
      }
      {
        std::lock_guard<std::mutex> lk(ctx->mu_);

        if (req->has_aggr_address()) {
          for (size_t i = 0; i < ctx->aggregators_.size(); ++i) {
            if (ctx->aggregators_[i].aggregator().empty()) {
              FragmentInfo info;

              info.set_id(i);
              info.set_aggregator(req->aggr_address());
              ctx->coordinator_->publish_aggregator(info);
              res->set_assigned_fragment(i);
              ctx->aggregators_[i] = info;

              break;
            }
          }
        }

        for (size_t i = 0; i < ctx->aggregators_.size(); ++i)
          if (!ctx->aggregators_[i].aggregator().empty())
            *res->add_aggregators() = ctx->aggregators_[i];
      }

      *res->mutable_options() = ctx->options_;
      Finish(Status::OK);
    }

    void OnDone(void) override { delete this; }
  };

  return new Reactor(this, req, res);
}
