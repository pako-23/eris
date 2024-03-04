#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.pb.h"
#include "algorithms/eris/split.h"
#include "erisfl/client.h"
#include "grpcpp/channel.h"
#include "grpcpp/server.h"
#include "zmq.hpp"

using grpc::Channel;
using grpc::Server;

struct AggregatorConfig {
  std::string address;
  uint16_t submit_port;
  uint16_t publish_port;
};

class ErisClient : public Client,
                   public std::enable_shared_from_this<ErisClient> {
public:
  explicit ErisClient(
      const std::string &coordinator_address,
      std::optional<AggregatorConfig> aggregator_opts = std::nullopt);
  void start(void) override;

private:
  bool start_aggregator(void);

  class ClientImpl {
  public:
    explicit ClientImpl(std::shared_ptr<Channel>, std::shared_ptr<ErisClient>);

    bool Join(void);

  private:
    std::unique_ptr<coordinator::Coordinator::Stub> stub_;
    std::shared_ptr<ErisClient> client_;
  };

  class AggregatorImpl {
  public:
    explicit AggregatorImpl(std::shared_ptr<ErisClient>);

  private:
    std::shared_ptr<ErisClient> client_;
  };

  const std::string coordinator_addr_;
  zmq::socket_t publisher_sock_;
  coordinator::TrainingOptions options_;
  const std::unique_ptr<SplitStrategy> splitter_;

  // Model publishing fields
  zmq::context_t zmq_context_;
  std::vector<std::string> aggregators_;
  std::vector<zmq::socket_t> subscriptions_;

  // Aggregation related fields
  std::unique_ptr<Server> aggregator_;
  const std::optional<AggregatorConfig> aggregator_config_;
};
