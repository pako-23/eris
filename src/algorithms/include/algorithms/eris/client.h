#pragma once

#include "algorithms/eris/aggregator.grpc.pb.h"
#include "algorithms/eris/aggregator.pb.h"
#include "algorithms/eris/builder.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "algorithms/eris/split.h"
#include "erisfl/client.h"
#include "grpcpp/channel.h"
#include "grpcpp/server.h"
#include "grpcpp/support/server_callback.h"
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// using grpc::Channel;
// using grpc::Server;

// class ErisClient : public Client,
//                    public std::enable_shared_from_this<ErisClient> {
// public:
//   explicit ErisClient(std::optional<ErisAggregatorBuilder> = std::nullopt);
//   void start(const std::string &) override;

// private:
//   void start_aggregator(const ErisAggregatorBuilder &);
//   void listen_coordinator_events(const std::string &);
//   bool aggregator_connect(const coordinator::FragmentInfo &);
//   void submit_parameters(uint32_t);
//   void receive_parameters(uint32_t);

//   class ClientImpl {
//   public:
//     explicit ClientImpl(std::shared_ptr<Channel>,
//     std::shared_ptr<ErisClient>);

//     bool Join(void);

//   private:
//     std::unique_ptr<coordinator::Coordinator::Stub> stub_;
//     std::shared_ptr<ErisClient> client_;
//   };

//   class AggregatorImpl final : public aggregator::Aggregator::CallbackService
//   { public:
//     explicit AggregatorImpl(const ErisAggregatorBuilder &);
//     grpc::ServerUnaryReactor *SubmitWeights(CallbackServerContext *,
//                                             const aggregator::Weight *,
//                                             aggregator::Empty *) override;

//   private:
//     uint32_t current_round_;
//     uint32_t min_clients_;
//     aggregator::WeightUpdate weight_update_;
//     zmq::context_t zmq_context_;
//     zmq::socket_t zmq_socket_;
//   };

//   zmq::context_t zmq_context_;

//   //  zmq::socket_t publisher_sock_;
//   std::unique_ptr<std::thread> coordinator_thread_;
//   coordinator::TrainingOptions options_;

//   // Parameter contribution fields
//   std::vector<std::unique_ptr<aggregator::Aggregator::Stub>> aggregators_;
//   std::vector<zmq::socket_t> subscriptions_;
//   std::mutex aggregation_mutex_;
//   uint32_t known_aggregators_;
//   std::condition_variable all_aggregators_connected_;

//   // Aggregation related fields
//   std::unique_ptr<std::thread> aggregator_thread_;
//   std::optional<ErisAggregatorBuilder> aggregator_builder_;

//   RandomSplit splitter_;
// };
