#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "grpcpp/support/server_callback.h"
#include "util/networking.h"
#include <cstddef>
#include <cstring>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/support/server_callback.h>
#include <grpcpp/support/status.h>
#include <mutex>
#include <vector>

using grpc::Status;
using grpc::StatusCode;

ErisCoordinator::ErisCoordinator(const ErisCoordinatorBuilder &builder)
    : service_{builder, this, builder.get_options()} {}

void ErisCoordinator::start(void) { service_.start(); }

void ErisCoordinator::stop(void) { service_.stop(); }

ErisCoordinator::CoordinatorImpl::CoordinatorImpl(
    ErisCoordinator *coordinator, const TrainingOptions &options)
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
              ctx->coordinator_->service_.publish(info);
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
