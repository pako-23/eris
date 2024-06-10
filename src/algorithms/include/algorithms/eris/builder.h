#pragma once

#include "algorithms/eris/coordinator.pb.h"
#include "util/networking.h"
#include <cstdint>
#include <string>

class ErisServiceBuilder {
public:
  explicit ErisServiceBuilder(void);

  bool add_rpc_port(uint16_t port);
  bool add_listen_address(const std::string &address);

  inline const std::string get_rpc_listen_address(void) const {
    return listen_address_ + ":" + std::to_string(rpc_port_);
  }

  inline const std::string get_rpc_public_address(void) const {
    return to_public_address(listen_address_) + ":" + std::to_string(rpc_port_);
  }

protected:
  std::string listen_address_;
  uint16_t rpc_port_;
};

class ErisCoordinatorBuilder final : public ErisServiceBuilder {
public:
  explicit ErisCoordinatorBuilder(void);

  bool add_rounds(uint32_t rounds);
  bool add_splits(uint32_t splits);
  bool add_min_clients(uint32_t min_clients);

  inline const coordinator::TrainingOptions &get_options(void) const {
    return options_;
  }

private:
  coordinator::TrainingOptions options_;
};

// class ErisAggregatorBuilder final : public ErisServiceBuilder {
// public:
//   explicit ErisAggregatorBuilder(void);

//   void add_min_clients(uint32_t);
//   void add_block_size(size_t);

//   inline uint32_t get_min_clients(void) const { return min_clients_; }
//   inline size_t get_block_size(void) const { return block_size_; }

// private:
//   size_t block_size_;
//   uint32_t min_clients_;
// };
