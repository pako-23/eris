#include "algorithms/eris/aggregator.h"
#include "algorithms/eris/builder.h"
#include "algorithms/eris/coordinator.h"
#include <algorithm>
#include <cstddef>
#include <grpcpp/server_builder.h>
#include <grpcpp/support/status.h>
#include <mutex>

using grpc::Status;
using grpc::StatusCode;

ErisAggregator::ErisAggregator(const ErisAggregatorBuilder &builder)
    : service_{builder, this, builder} {}

void ErisAggregator::start(void) noexcept { service_.start(); }

void ErisAggregator::stop(void) noexcept { service_.stop(); }

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
        for (const float val : ctx->weights_)
          update.add_weight(val);

        ctx->aggregator_->service_.publish(update);
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
