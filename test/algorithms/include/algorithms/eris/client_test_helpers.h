#pragma once

#include "algorithms/eris/builder.h"
#include "algorithms/eris/client.h"
#include "algorithms/eris/common.pb.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "algorithms/eris/service.h"
#include <bits/types/struct_sched_param.h>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <grpcpp/server_context.h>
#include <grpcpp/support/server_callback.h>
#include <grpcpp/support/status.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>

using eris::InitialState;
using eris::JoinRequest;
using eris::TrainingOptions;
using grpc::CallbackServerContext;

class ErisMockClient : public ErisClient {
public:
  explicit ErisMockClient(size_t parameters_size)
      : ErisClient{}, parameters_size_{parameters_size} {}

  std::vector<double> get_parameters(void) const {
    std::default_random_engine rng(time(NULL));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> weigths(parameters_size_);

    for (size_t i = 0; i < parameters_size_; ++i)
      weigths[i] = dist(rng);

    return weigths;
  }
  void set_parameters(const std::vector<double> &parameters) {}
  void fit(void) {}

private:
  size_t parameters_size_;
};

class MockAggregator final {
public:
  explicit MockAggregator(void)
      : service_{MockAggregatorBuilder{}, this}, received{0} {

    thread_ = std::make_unique<std::thread>([this]() { service_.start(); });

    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
        get_rpc_address(), grpc::InsecureChannelCredentials());

    bool connected = channel->WaitForConnected(
        std::chrono::system_clock::now() + std::chrono::minutes(1));

    if (!connected)
      throw std::runtime_error{"failed to setup mock coordinator"};
  }

  ~MockAggregator(void) {
    service_.stop();
    thread_->join();
  }

  inline std::string get_rpc_address(void) const {
    return "127.0.0.1:" + std::to_string(service_.get_rpc_port());
  }

  inline std::string get_pubsub_address(void) const {
    return "tcp://127.0.0.1:" + std::to_string(service_.get_publish_port());
  }

  inline size_t get_received(void) const { return received; }

  inline void publish_update(const eris::WeightUpdate &update) {
    service_.publish(update);
  }

private:
  class MockAggregatorBuilder : public ErisServiceBuilder {
  public:
    explicit MockAggregatorBuilder(void) : ErisServiceBuilder{} {
      add_rpc_port(0);
      add_publish_port(0);
    }
  };

  class MockAggregatorService : public eris::Aggregator::CallbackService {
  public:
    explicit MockAggregatorService(MockAggregator *aggr) noexcept
        : aggr_{aggr} {}

    grpc::ServerUnaryReactor *SubmitWeights(CallbackServerContext *ctx,
                                            const FragmentWeights *req,
                                            eris::Empty *res) override {
      class Reactor : public grpc::ServerUnaryReactor {
      public:
        explicit Reactor(MockAggregator *aggr) {
          ++aggr->received;
          Finish(grpc::Status::OK);
        }

        void OnDone(void) override { delete this; }
      };

      return new Reactor(aggr_);
    }

  private:
    MockAggregator *aggr_;
  };

  ErisService<MockAggregatorService> service_;
  std::unique_ptr<std::thread> thread_;

  std::atomic_size_t received;
};

class MockCoordinator {
public:
  MockCoordinator(uint32_t min_clients, uint32_t rounds, uint32_t split_seed,
                  uint32_t splits)
      : thread_{nullptr}, service_{MockCoordinatorBuilder{}, this},
        fail_request_{false}, aggregators_{} {
    options_.set_min_clients(min_clients);
    options_.set_rounds(rounds);
    options_.set_split_seed(split_seed);
    options_.set_splits(splits);
    aggregators_.resize(options_.splits());

    thread_ = std::make_unique<std::thread>([this]() { service_.start(); });

    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
        get_rpc_address(), grpc::InsecureChannelCredentials());

    bool connected = channel->WaitForConnected(
        std::chrono::system_clock::now() + std::chrono::minutes(1));

    if (!connected)
      throw std::runtime_error{"failed to setup mock coordinator"};
  }

  ~MockCoordinator(void) {
    service_.stop();
    thread_->join();
  }

  inline std::string get_rpc_address(void) const {
    return "127.0.0.1:" + std::to_string(service_.get_rpc_port());
  }

  inline std::string get_pubsub_address(void) const {
    return "tcp://127.0.0.1:" + std::to_string(service_.get_publish_port());
  }

  void set_fail_requests(bool fail) { fail_request_ = true; }

  void add_aggregator(uint32_t id, const std::string &submit_address,
                      const std::string &publish_address) {
    if (id >= options_.splits())
      return;

    FragmentInfo info;

    info.set_id(id);
    info.set_submit_address(submit_address);
    info.set_publish_address(publish_address);

    aggregators_[id] = std::make_optional<FragmentInfo>(info);
  }

  inline const std::optional<FragmentInfo> &get_aggregator(size_t i) const {
    return aggregators_[i];
  }

  inline const TrainingOptions &get_options(void) const { return options_; }

private:
  class MockCoordinatorBuilder : public ErisServiceBuilder {
  public:
    explicit MockCoordinatorBuilder(void) : ErisServiceBuilder{} {
      add_rpc_port(0);
      add_publish_port(0);
    }
  };

  class MockCoordinatorService : public eris::Coordinator::CallbackService {
  public:
    explicit MockCoordinatorService(MockCoordinator *coordinator)
        : coordinator_{coordinator} {}

    grpc::ServerUnaryReactor *Join(CallbackServerContext *ctx,
                                   const JoinRequest *req,
                                   InitialState *res) override {
      class Reactor : public grpc::ServerUnaryReactor {
      public:
        explicit Reactor(MockCoordinator *coordinator, const JoinRequest *req,
                         InitialState *res) {
          if (coordinator->fail_request_) {
            Finish(grpc::Status(grpc::StatusCode::INTERNAL, "Error"));
            return;
          }

          *res->mutable_options() = coordinator->options_;
          for (const auto &aggr : coordinator->aggregators_)
            if (aggr)
              *res->add_aggregators() = *aggr;

          if (req->has_submit_address()) {
            for (uint32_t i = 0; i < coordinator->aggregators_.size(); ++i)
              if (!coordinator->aggregators_[i]) {
                FragmentInfo info;

                info.set_submit_address(req->submit_address());
                info.set_id(i);
                info.set_publish_address(req->publish_address());

                coordinator->aggregators_[i] =
                    std::make_optional<FragmentInfo>(info);

                res->set_assigned_fragment(i);
                *res->add_aggregators() = info;
                coordinator->service_.publish(info);
                break;
              }
          }

          Finish(grpc::Status::OK);
        }

        void OnDone(void) override { delete this; }
      };

      return new Reactor(coordinator_, req, res);
    };

  private:
    MockCoordinator *coordinator_;
  };

  std::unique_ptr<std::thread> thread_;
  ErisService<MockCoordinatorService> service_;
  TrainingOptions options_;
  bool fail_request_;
  std::vector<std::optional<FragmentInfo>> aggregators_;
};
