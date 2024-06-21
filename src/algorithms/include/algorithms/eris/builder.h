#pragma once

#include "algorithms/eris/coordinator.pb.h"
#include <cstdint>
#include <string>

#define DEFAULT_ERIS_PORT 50051

class ErisServiceBuilder {
public:
  explicit ErisServiceBuilder(void);
  virtual ~ErisServiceBuilder(void) = default;
  bool add_rpc_port(uint16_t port);
  bool add_listen_address(const std::string &address);

  inline const std::string get_rpc_listen_address(void) const {
    return listen_address_ + ":" + std::to_string(rpc_port_);
  }

protected:
  std::string listen_address_;
  uint16_t rpc_port_;
};

class ErisCoordinatorBuilder final : public ErisServiceBuilder {
public:
  explicit ErisCoordinatorBuilder(void);
  virtual ~ErisCoordinatorBuilder(void) = default;

  bool add_rounds(uint32_t rounds);
  bool add_splits(uint32_t splits);
  bool add_min_clients(uint32_t min_clients);
  bool add_split_seed(uint32_t split_seed);

  inline const eris::TrainingOptions &get_options(void) const {
    return options_;
  }

private:
  eris::TrainingOptions options_;
};
