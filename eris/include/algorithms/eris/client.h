#pragma once

#include "algorithms/eris/aggregator.grpc.pb.h"
#include "algorithms/eris/aggregator.pb.h"
#include "algorithms/eris/builder.h"
#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "erisfl/client.h"
#include "grpcpp/channel.h"
#include "grpcpp/server.h"
#include "grpcpp/support/server_callback.h"
#include "zmq.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using grpc::Channel;
using grpc::Server;

class ErisClient : public Client,
                   public std::enable_shared_from_this<ErisClient> {
public:
  explicit ErisClient(std::optional<ErisAggregatorBuilder> = std::nullopt);
  void start(const std::string &) override;

private:
  void start_aggregator(const ErisAggregatorBuilder &);

  class ClientImpl {
  public:
    explicit ClientImpl(std::shared_ptr<Channel>, std::shared_ptr<ErisClient>);

    bool Join(void);

  private:
    std::unique_ptr<coordinator::Coordinator::Stub> stub_;
    std::shared_ptr<ErisClient> client_;
  };

  class AggregatorImpl final : public aggregator::Aggregator::CallbackService {
  public:
    explicit AggregatorImpl(const ErisAggregatorBuilder &);
    grpc::ServerUnaryReactor *SubmitWeights(CallbackServerContext *,
                                            const aggregator::Weight *,
                                            aggregator::Empty *) override;

  private:
    uint32_t current_round_;
    uint32_t min_clients_;
    aggregator::WeightUpdate weight_update_;
    zmq::context_t zmq_context_;
    zmq::socket_t zmq_socket_;
  };

  zmq::context_t zmq_context_;
  zmq::socket_t publisher_sock_;
  coordinator::TrainingOptions options_;

  // Model publishing fields
  std::vector<std::string> aggregators_;
  std::vector<zmq::socket_t> subscriptions_;

  // Aggregation related fields
  std::unique_ptr<std::thread> aggregator_;
  std::optional<ErisAggregatorBuilder> aggregator_builder_;
};
