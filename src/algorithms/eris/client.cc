#include "algorithms/eris/client.h"
#include "algorithms/eris/aggregator.grpc.pb.h"
#include "algorithms/eris/aggregator.h"
#include "algorithms/eris/builder.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.pb.h"
#include "spdlog/spdlog.h"
#include "util/networking.h"
#include "zmq.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/support/channel_arguments.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/stub_options.h>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

ErisClient::ErisClient(void)
    : rpc_address_{}, subscribe_address_{}, aggr_address_{}, aggr_rpc_port_{0},
      state_{} {}

bool ErisClient::train(void) {
  if (rpc_address_.empty()) {
    spdlog::error("missing coordinator RPC address");
    return false;
  }

  if (subscribe_address_.empty()) {
    spdlog::error("missing coordinator publish address");
    return false;
  }

  bool joined = false;
  if (aggr_address_.empty())
    joined = state_.join(this, rpc_address_, subscribe_address_);
  else
    joined = state_.join(this, rpc_address_, subscribe_address_, &aggr_address_,
                         &aggr_rpc_port_, &aggr_publish_port_);

  if (!joined) {
    spdlog::error("failed to join the training");
    return false;
  }

  std::this_thread::sleep_for(std::chrono::seconds(2));

  uint32_t round = 0;

  while (round != state_.get_options().rounds()) {
    fit();

    if (!state_.submit_weights(get_parameters(), round))
      return false;

    set_parameters(state_.receive_weights(&round));
    evaluate();
    spdlog::info("finished with round {0}", round);
    ++round;
  }

  return true;
}

bool ErisClient::set_coordinator_rpc(const std::string &address) {
  if (!valid_aggregator_submit(address)) {
    spdlog::error("the provided coordinator RPC address is not valid");
    return false;
  }

  rpc_address_ = address;
  return true;
}

bool ErisClient::set_coordinator_subscription(const std::string &address) {
  if (!valid_aggregator_publish(address)) {
    spdlog::error("the provided coordinator subscription address is not valid");
    return false;
  }

  subscribe_address_ = address;
  return true;
}

bool ErisClient::set_aggregator_config(const std::string &address,
                                       uint16_t submit_port,
                                       uint16_t publish_port) {
  if (!valid_ipv4(address) || address == "0.0.0.0" || submit_port == 0 ||
      publish_port == 0 || submit_port == publish_port) {
    spdlog::error("the provided aggregator configuration is not valid");
    return false;
  }

  aggr_address_ = address;
  aggr_rpc_port_ = submit_port;
  aggr_publish_port_ = publish_port;
  return true;
}

ErisClient::ClientState::ClientState(void)
    : coord_subscribed_{false}, mu_{}, splitter(), coord_updater_{nullptr},
      aggregator_{nullptr}, aggregator_thread_{nullptr} {
  zmq_ctx = zmq_ctx_new();
  if (!zmq_ctx)
    throw std::bad_alloc{};

  coord_sub = zmq_socket(zmq_ctx, ZMQ_SUB);
  if (!coord_sub)
    throw std::bad_alloc{};
}

ErisClient::ClientState::~ClientState(void) {
  if (aggregator_)
    aggregator_->stop();
  if (aggregator_thread_)
    aggregator_thread_->join();
  coord_subscribed_ = false;
  if (coord_updater_)
    coord_updater_->join();

  zmq_close(coord_sub);

  for (void *sub : subscriptions_)
    if (sub)
      zmq_close(sub);

  zmq_ctx_destroy(zmq_ctx);
}

bool ErisClient::ClientState::join(ErisClient *client,
                                   const std::string &rpc_address,
                                   const std::string &subscribe_address,
                                   const std::string *listen_address,
                                   const uint16_t *rpc_port,
                                   const uint16_t *publish_port) {
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(rpc_address, grpc::InsecureChannelCredentials());
  std::unique_ptr<eris::Coordinator::Stub> stub =
      eris::Coordinator::NewStub(channel);

  grpc::ClientContext ctx;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  bool success = true;
  eris::JoinRequest req;
  eris::InitialState res;

  if (listen_address) {
    req.set_submit_address(*listen_address + ":" + std::to_string(*rpc_port));
    req.set_publish_address("tcp://" + *listen_address + ":" +
                            std::to_string(*publish_port));
  }

  stub->async()->Join(
      &ctx, &req, &res,
      [this, client, &mu, &done, &cv, &success, &res](grpc::Status s) {
        if (!s.ok() || !configure(client, res)) {
          std::lock_guard<std::mutex> lk(mu);
          done = true;
          success = false;
          cv.notify_one();
          return;
        }

        std::lock_guard<std::mutex> lk(mu);
        done = true;
        cv.notify_one();
      });

  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [&done] { return done; });

  if (!success)
    return success;

  coordinator_subscribe(subscribe_address);

  if (res.has_assigned_fragment())
    start_aggregator(res.assigned_fragment(), listen_address, rpc_port,
                     publish_port);

  return success;
}

bool ErisClient::ClientState::configure(ErisClient *client,
                                        const InitialState &state) {
  {
    std::lock_guard<std::mutex> state_lk(mu_);
    subscriptions_.resize(state.options().splits());
    submitters_.resize(state.options().splits());

    for (const auto &aggregator : state.aggregators())
      if (!register_aggregator(aggregator))
        return false;
  }

  options_ = state.options();
  splitter.configure(client->get_parameters(), state.options().splits(),
                     state.options().split_seed());

  return true;
}

bool ErisClient::ClientState::submit_weights(
    const std::vector<float> &parameters, uint32_t round) {
  auto fragments = splitter.split(parameters, round);
  const int max_retries = 3;
  std::vector<eris::Empty> res(fragments.size());
  std::vector<bool> done(fragments.size(), false);

  std::mutex mu;
  std::condition_variable cv;
  bool success = true;
  std::atomic_size_t finished = 0;
  std::atomic_size_t failed = 0;

  for (int attempt = 1; attempt <= max_retries; ++attempt) {
    std::vector<grpc::ClientContext> ctx(fragments.size());

    for (size_t i = 0; i < fragments.size(); ++i) {
      if (done[i])
        continue;

      std::unique_lock<std::mutex> lk(mu_);
      if (!submitters_[i])
        aggregator_joined_.wait(
            lk, [this, &i]() { return submitters_[i] != nullptr; });

      submitters_[i]->async()->SubmitWeights(
          &ctx[i], &fragments[i], &res[i],
          [&finished, i, &done, &failed, &mu, &cv, &success](grpc::Status s) {
            if (!s.ok() && s.error_code() == grpc::INVALID_ARGUMENT) {
              std::lock_guard<std::mutex> lk(mu);
              success = false;
              done[i] = true;
              ++finished;
            } else if (!s.ok())
              ++failed;
            else {
              done[i] = true;
              ++finished;
            }

            cv.notify_one();
          });
    }

    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&failed, &finished, &fragments] {
      return finished + failed == fragments.size();
    });

    if (failed == 0)
      return success;

    for (std::vector<bool>::size_type i = 0; i < done.size(); ++i)
      if (!done[i])
        spdlog::warn("failed to submit weights for fragment {0} at attempt {1}",
                     i, attempt);

    failed = 0;
    std::this_thread::sleep_for(attempt * std::chrono::seconds(1));
  }

  for (std::vector<bool>::size_type i = 0; i < done.size(); ++i)
    if (!done[i])
      spdlog::error("failed to submit weights for fragment {0}", i);

  return false;
}

std::vector<float> ErisClient::ClientState::receive_weights(uint32_t *round) {
  std::vector<WeightUpdate> weights(options_.splits());

  std::vector<bool> done(options_.splits(), false);
  size_t i = 0;

  while (i < subscriptions_.size()) {
    zmq_msg_t msg;

    if (done[i]) {
      ++i;
      continue;
    }

    zmq_msg_init(&msg);

    int size = zmq_msg_recv(&msg, subscriptions_[i], 0);
    if (size <= 0) {
      if (query_fragment(submitters_[i], &weights[i]))
        goto handle_weight;

      spdlog::error("failed to receive weights from aggregator {0}", i);
      zmq_msg_close(&msg);
      continue;
    }
    weights[i].ParseFromArray(zmq_msg_data(&msg), size);
    zmq_msg_close(&msg);

  handle_weight:
    if (weights[i].round() == *round) {
      done[i] = true;
      ++i;
    } else if (weights[i].round() > *round) {
      *round = weights[i].round();
      std::fill(done.begin(), done.end(), false);
      done[i] = true;
      i = 0;
    }
  }

  return splitter.reassemble(weights);
}

bool ErisClient::ClientState::register_aggregator(
    const FragmentInfo &aggregator) noexcept {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      aggregator.submit_address(), grpc::InsecureChannelCredentials());
  submitters_[aggregator.id()] = eris::Aggregator::NewStub(channel);
  subscriptions_[aggregator.id()] = zmq_socket(zmq_ctx, ZMQ_SUB);
  if (subscriptions_[aggregator.id()] == nullptr ||
      zmq_connect(subscriptions_[aggregator.id()],
                  aggregator.publish_address().c_str()) != 0)
    return false;
  int zmq_timeout = 5000;

  zmq_setsockopt(subscriptions_[aggregator.id()], ZMQ_SUBSCRIBE, "", 0);
  zmq_setsockopt(subscriptions_[aggregator.id()], ZMQ_RCVTIMEO, &zmq_timeout,
                 sizeof(zmq_timeout));
  aggregator_joined_.notify_all();
  return true;
}

void ErisClient::ClientState::start_aggregator(
    uint32_t fragment_id, const std::string *listen_address,
    const uint16_t *rpc_port, const uint16_t *publish_port) noexcept {
  ErisAggregatorBuilder builder{fragment_id,
                                splitter.get_fragment_size(fragment_id)};

  builder.add_rpc_listen_address(*listen_address);
  builder.add_publish_address(*listen_address);
  builder.add_publish_port(*publish_port);
  builder.add_rpc_port(*rpc_port);
  builder.add_min_clients(options_.min_clients());

  aggregator_ = std::make_unique<ErisAggregator>(builder);
  aggregator_thread_ = std::make_unique<std::thread>(
      [](ErisAggregator *aggregator) { aggregator->start(); },
      aggregator_.get());
}

void ErisClient::ClientState::coordinator_subscribe(
    const std::string &subscribe_address) noexcept {
  int zmq_timeout = 500;

  coord_subscribed_ = true;
  zmq_connect(coord_sub, subscribe_address.c_str());
  zmq_setsockopt(coord_sub, ZMQ_SUBSCRIBE, "", 0);
  zmq_setsockopt(coord_sub, ZMQ_RCVTIMEO, &zmq_timeout, sizeof(zmq_timeout));

  coord_updater_ = std::make_unique<std::thread>([this]() {
    FragmentInfo aggregator;

    while (coord_subscribed_) {
      zmq_msg_t msg;

      zmq_msg_init(&msg);

      int size = zmq_msg_recv(&msg, coord_sub, 0);
      if (size > 0) {
        std::lock_guard<std::mutex> lk(mu_);
        aggregator.ParseFromArray(zmq_msg_data(&msg), size);
        register_aggregator(aggregator);
        zmq_msg_close(&msg);
      }
    }
  });
}

bool ErisClient::ClientState::query_fragment(
    std::unique_ptr<eris::Aggregator::Stub> &aggr,
    WeightUpdate *update) noexcept {
  grpc::ClientContext ctx;
  Empty req;
  std::mutex mu;
  bool done = false;
  bool success = true;
  std::condition_variable cv;

  aggr->async()->GetUpdate(&ctx, &req, update,
                           [&success, &done, &mu, &cv](grpc::Status s) {
                             if (!s.ok()) {
                               std::lock_guard<std::mutex> lk(mu);
                               done = true;
                               success = false;
                               cv.notify_one();
                               return;
                             }
                             std::lock_guard<std::mutex> lk(mu);
                             done = true;
                             cv.notify_one();
                           });

  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [&done] { return done; });

  return success && update->contributors() >= options_.min_clients();
}
