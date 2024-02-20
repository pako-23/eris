#pragma once

#include <cstdint>
#include <memory>

#include "algorithms/eris/coordinator.grpc.pb.h"
#include "erisfl/client.h"
#include "grpcpp/channel.h"

using grpc::Channel;

class ErisClient : public Client {
 public:
  explicit ErisClient(const std::string& coordinator_address,
                      const std::string& address = "0.0.0.0",
                      uint16_t port = 50051);
  void run(void) override;

 private:
  class ClientImpl {
   public:
    explicit ClientImpl(std::shared_ptr<Channel>);

    bool Join(const std::string&, uint16_t);

   private:
    std::unique_ptr<coordinator::Coordinator::Stub> stub_;
  };

  const std::string bind_addr_;
  const uint16_t bind_port_;
  const std::string coordinator_addr_;
};
