#include "algorithms/eris/client.h"
#include "algorithms/eris/aggregator.grpc.pb.h"
#include "algorithms/eris/aggregator.h"
#include "algorithms/eris/builder.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.pb.h"
#include "util/networking.h"
#include "zmq.h"
#include <condition_variable>
#include <cstdint>
#include <grpcpp/channel.h>
#include <grpcpp/create_channel.h>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

ErisClient::ErisClient(void)
    : aggregator_{nullptr}, aggregator_thread_{nullptr}, rpc_address_{},
      subscribe_address_{}, aggr_address_{}, aggr_rpc_port_{0},
      aggr_publish_port_{0}, coord_updater_{nullptr}, splitter() {
  zmq_ctx = zmq_ctx_new();
  if (!zmq_ctx)
    throw std::bad_alloc{};

  coord_sub = zmq_socket(zmq_ctx, ZMQ_SUB);
  if (!coord_sub)
    throw std::bad_alloc{};
}

ErisClient::~ErisClient(void) {
  if (aggregator_)
    aggregator_->stop();
  if (aggregator_thread_)
    aggregator_thread_->join();
  listening_ = false;
  if (coord_updater_)
    coord_updater_->join();

  zmq_close(coord_sub);

  for (void *sub : subscriptions_)
    if (sub)
      zmq_close(sub);

  zmq_ctx_destroy(zmq_ctx);
}

void ErisClient::start(void) {}

bool ErisClient::set_coordinator_rpc(const std::string &address) {
  if (!valid_aggregator_submit(address))
    return false;

  rpc_address_ = address;
  return true;
}

bool ErisClient::set_coordinator_subscription(const std::string &address) {
  if (!valid_aggregator_publish(address))
    return false;
  subscribe_address_ = address;
  return true;
}

bool ErisClient::set_aggregator_config(const std::string &address,
                                       uint16_t submit_port,
                                       uint16_t publish_port) {

  if (!valid_ipv4(address) || address == "0.0.0.0" || submit_port == 0 ||
      publish_port == 0)
    return false;

  aggr_address_ = address;
  aggr_rpc_port_ = submit_port;
  aggr_publish_port_ = publish_port;
  return true;
}

bool ErisClient::join(void) {
  if (rpc_address_.empty() || subscribe_address_.empty())
    return false;

  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(rpc_address_, grpc::InsecureChannelCredentials());
  std::unique_ptr<eris::Coordinator::Stub> stub =
      eris::Coordinator::NewStub(channel);

  grpc::ClientContext ctx;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  bool success = true;
  eris::JoinRequest req;
  eris::InitialState res;

  if (!aggr_address_.empty()) {
    req.set_submit_address(aggr_address_ + ":" +
                           std::to_string(aggr_rpc_port_));
    req.set_publish_address("tcp://" + aggr_address_ + ":" +
                            std::to_string(aggr_publish_port_));
  }

  stub->async()->Join(&ctx, &req, &res,
                      [this, &mu, &done, &cv, &success, &res](grpc::Status s) {
                        if (!s.ok()) {
                          std::lock_guard<std::mutex> lk(mu);
                          done = true;
                          success = false;
                          cv.notify_one();
                          return;
                        }

                        {
                          std::lock_guard<std::mutex> state_lk(mu_);
                          subscriptions_.resize(res.options().splits());
                          submitters_.resize(res.options().splits());

                          for (const auto &aggregator : res.aggregators()) {
                            if (!register_aggregator(aggregator)) {
                              std::lock_guard<std::mutex> lk(mu);
                              done = true;
                              success = false;
                              cv.notify_one();
                              return;
                            }
                          }
                        }

                        options_ = res.options();
                        splitter.configure(get_parameters(),
                                           res.options().splits(),
                                           res.options().split_seed());

                        std::lock_guard<std::mutex> lk(mu);
                        done = true;
                        cv.notify_one();
                      });

  std::unique_lock<std::mutex> lk(mu);
  cv.wait(lk, [&done] { return done; });

  if (!success)
    return success;

  coordinator_subscribe();

  if (res.has_assigned_fragment())
    start_aggregator(res.assigned_fragment());

  return success;
}

void ErisClient::start_aggregator(uint32_t fragment_id) noexcept {
  ErisAggregatorBuilder builder{fragment_id,
                                splitter.get_fragment_size(fragment_id)};

  builder.add_rpc_listen_address(aggr_address_);
  builder.add_publish_address(aggr_address_);
  builder.add_publish_port(aggr_publish_port_);
  builder.add_rpc_port(aggr_rpc_port_);
  builder.add_min_clients(options_.min_clients());

  aggregator_ = std::make_unique<ErisAggregator>(builder);
  aggregator_thread_ = std::make_unique<std::thread>(
      [](ErisAggregator *aggregator) { aggregator->start(); },
      aggregator_.get());
}

bool ErisClient::register_aggregator(const FragmentInfo &aggregator) noexcept {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      aggregator.submit_address(), grpc::InsecureChannelCredentials());
  submitters_[aggregator.id()] = eris::Aggregator::NewStub(channel);
  subscriptions_[aggregator.id()] = zmq_socket(zmq_ctx, ZMQ_SUB);
  return subscriptions_[aggregator.id()] != nullptr &&
         zmq_connect(subscriptions_[aggregator.id()],
                     aggregator.publish_address().c_str()) == 0;
}

void ErisClient::coordinator_subscribe(void) noexcept {
  coord_updater_ = std::make_unique<std::thread>([this]() {
    FragmentInfo aggregator;
    zmq_msg_t msg;
    int zmq_timeout = 500;

    listening_ = true;
    zmq_connect(coord_sub, subscribe_address_.c_str());
    zmq_setsockopt(coord_sub, ZMQ_SUBSCRIBE, "", 0);
    zmq_setsockopt(coord_sub, ZMQ_RCVTIMEO, &zmq_timeout, sizeof(zmq_timeout));
    zmq_msg_init(&msg);

    while (listening_) {
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
