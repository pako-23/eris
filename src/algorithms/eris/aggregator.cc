#include "algorithms/eris/aggregator.h"
#include "algorithms/eris/builder.h"
#include "algorithms/eris/coordinator.h"
#include "spdlog/spdlog.h"
#include "zmq.h"
#include <algorithm>
#include <cstddef>
#include <grpcpp/server_builder.h>
#include <grpcpp/support/status.h>
#include <mutex>

using grpc::Status;
using grpc::StatusCode;

ErisAggregator::ErisAggregator(const ErisAggregatorBuilder &builder)
    : server_{nullptr}, service_{this, builder}, started_{false},
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

  publish_port_ = atoi(strchr(strchr(endpoint, ':') + 1, ':') + 1);

  srv_builder.AddListeningPort(builder.get_rpc_listen_address(),
                               grpc::InsecureServerCredentials(), &port);
  srv_builder.RegisterService(&service_);

  server_ = srv_builder.BuildAndStart();
  rpc_port_ = port;
}

ErisAggregator::~ErisAggregator(void) {
  if (started_)
    stop();
  zmq_close(publisher);
  zmq_ctx_destroy(zmq_ctx);
}

void ErisAggregator::start(void) {
  started_ = true;
  char endpoint[255];
  size_t endpointlen = sizeof(endpoint);

  zmq_getsockopt(publisher, ZMQ_LAST_ENDPOINT, &endpoint, &endpointlen);

  spdlog::info("listening RPC requests on {0}:{1} and publishing on {2}",
               listening_address_.substr(0, listening_address_.find(':')),
               rpc_port_, endpoint);
  server_->Wait();
}

void ErisAggregator::stop(void) {
  server_->Shutdown();
  started_ = false;
}

bool ErisAggregator::publish_weight(const WeightUpdate &update) {
  zmq_msg_t msg;

  zmq_msg_init_size(&msg, update.ByteSizeLong());
  update.SerializeToArray(zmq_msg_data(&msg), update.ByteSizeLong());
  bool ret = zmq_msg_send(&msg, publisher, 0) > 0;
  zmq_msg_close(&msg);

  return ret;
}

ErisAggregator::AggregatorImpl::AggregatorImpl(
    ErisAggregator *aggregator, const ErisAggregatorBuilder &builder) noexcept
    : round_{0}, weights_(builder.get_fragment_size(), 0.0), contributors_{0},
      mu_{}, fragment_size_{builder.get_fragment_size()},
      min_clients_{builder.get_min_client()}, aggregator_{aggregator} {}

grpc::ServerUnaryReactor *ErisAggregator::AggregatorImpl::SubmitWeights(
    CallbackServerContext *ctx, const FragmentWeights *req, eris::Empty *res) {
  class Reactor : public grpc::ServerUnaryReactor {
  public:
    explicit Reactor(AggregatorImpl *ctx, const FragmentWeights *req) {
      if ((size_t)req->weight_size() != ctx->fragment_size_) {
        Finish(Status(StatusCode::INVALID_ARGUMENT, "Wrong fragment size"));
        return;
      }

      std::lock_guard lk(ctx->mu_);
      if (req->round() != ctx->round_) {
        Finish(Status(StatusCode::INVALID_ARGUMENT, "Wrong round number"));
        return;
      }

      for (int i = 0; i < req->weight_size(); ++i)
        ctx->weights_[i] += req->weight(i);

      if (++ctx->contributors_ == ctx->min_clients_) {
        WeightUpdate update;

        update.set_round(ctx->round_);
        update.set_contributors(ctx->contributors_);
        for (const double val : ctx->weights_)
          update.add_weight(val);

        ctx->aggregator_->publish_weight(update);
        std::fill(ctx->weights_.begin(), ctx->weights_.end(), 0);
        ctx->contributors_ = 0;
        ++ctx->round_;
      }

      Finish(Status::OK);
    }

    void OnDone(void) override { delete this; }
  };

  return new Reactor(this, req);
}
