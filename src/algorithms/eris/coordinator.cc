#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include "spdlog/spdlog.h"
#include "util/networking.h"
#include <cstddef>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/support/server_callback.h>
#include <grpcpp/support/status.h>
#include <memory>
#include <mutex>
#include <vector>

ErisCoordinator::ErisCoordinator(const ErisCoordinatorBuilder &builder)
    : server_{nullptr}, service_{builder.get_options()}, started_{false},
      listening_address_{builder.get_rpc_listen_address()} {
  grpc::ServerBuilder srv_builder;
  int port;

  srv_builder.AddListeningPort(builder.get_rpc_listen_address(),
                               grpc::InsecureServerCredentials(), &port);
  srv_builder.RegisterService(&service_);

  server_ = srv_builder.BuildAndStart();
  listening_port_ = port;
}

ErisCoordinator::~ErisCoordinator(void) {
  if (started_)
    stop();
}

void ErisCoordinator::start(void) {
  started_ = false;
  spdlog::info("started coordinator on {0}:{1}",
               listening_address_.substr(0, listening_address_.find(':')),
               listening_port_);
  server_->Wait();
}

void ErisCoordinator::stop(void) {
  server_->Shutdown();
  started_ = false;
}

ErisCoordinator::Aggregators::Aggregators(size_t splits)
    : aggregators_(splits), mu_{} {}

void ErisCoordinator::Aggregators::wait_update(void) {
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk);
}

void ErisCoordinator::Aggregators::handle_join_request(const JoinRequest *req,
                                                       InitialUpdate *update) {
  std::lock_guard<std::mutex> lk(mu_);

  if (req->has_aggr_address()) {
    for (size_t i = 0; i < aggregators_.size(); ++i) {
      if (aggregators_[i].aggregator().empty()) {
        update->set_assigned_fragment(i);
        aggregators_[i].set_id(i);
        aggregators_[i].set_aggregator(req->aggr_address());
        break;
      }
    }
  }

  for (size_t i = 0; i < aggregators_.size(); ++i)
    if (!aggregators_[i].aggregator().empty())
      *update->add_aggregators() = aggregators_[i];
}

ErisCoordinator::CoordinatorImpl::CoordinatorImpl(
    const TrainingOptions &options)
    : options_{options}, aggregators_(options.splits()) {}

grpc::ServerWriteReactor<CoordinatorUpdate> *
ErisCoordinator::CoordinatorImpl::Join(CallbackServerContext *ctx,
                                       const JoinRequest *req) {

  class Updater : public grpc::ServerWriteReactor<CoordinatorUpdate> {
  public:
    explicit Updater(const TrainingOptions &options, const JoinRequest *req,
                     Aggregators &aggregators)
        : options_{options}, aggregators_{aggregators} {
      if (req->has_aggr_address() && !valid_aggregator(req->aggr_address())) {
        Finish(
            Status(StatusCode::INVALID_ARGUMENT,
                   "An aggregator address must have the form <address>:<port> "
                   "where address is a valid IPv4 address"));
        return;
      }

      InitialUpdate *update = res_.mutable_init_config();
      *update->mutable_options() = options_;

      aggregators_.handle_join_request(req, update);

      StartWrite(&res_);
    }

    void OnDone(void) override { delete this; }

    void OnWriteDone([[maybe_unused]] bool ok) override { Finish(Status::OK); }

  private:
    CoordinatorUpdate res_;
    const TrainingOptions &options_;
    Aggregators &aggregators_;
  };

  return new Updater(options_, req, aggregators_);
}
