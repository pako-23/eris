#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "algorithms/eris/coordinator.grpc.pb.h"
#include "algorithms/eris/coordinator.pb.h"
#include "erisfl/client.h"
#include "grpcpp/channel.h"
#include "zmq.hpp"

using grpc::Channel;

class ErisClient : public Client,
                   public std::enable_shared_from_this<ErisClient> {
public:
  explicit ErisClient(const std::string &coordinator_address,
                      const std::string &address = "0.0.0.0",
                      uint16_t grpc_port = 50051,
                      std::optional<uint16_t> pub_port = std::nullopt);
  void start(void) override;

private:
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

  const std::string bind_addr_;
  const uint16_t grpc_port_;
  const std::optional<uint16_t> pub_port_;
  const std::string coordinator_addr_;
  std::vector<coordinator::Endpoint> aggregators_;
  coordinator::TrainingOptions options_;

  zmq::context_t publisher_ctx_;
  zmq::socket_t publisher_sock_;
};
