#pragma once

#include "algorithms/eris/coordinator.pb.h"
#include <cstddef>
#include <cstdint>
#include <string>

class ErisServiceBuilder {
public:
  explicit ErisServiceBuilder(void);

  void add_publish_port(uint16_t);
  void add_rpc_port(uint16_t);
  void add_listen_address(const std::string &);

  const std::string get_rpc_address(void) const;
  const std::string get_zmq_listen_address(void) const;
  const std::string get_zmq_publish_address(void) const;

protected:
  std::string listen_address_;
  uint16_t publish_port_;
  uint16_t rpc_port_;
};

class ErisCoordinatorBuilder final : public ErisServiceBuilder {
public:
  explicit ErisCoordinatorBuilder(void);

  void add_rounds(uint32_t);
  void add_splits(uint32_t);
  void add_min_clients(uint32_t);

private:
  friend class ErisCoordinator;

  coordinator::TrainingOptions options_;
};

class ErisAggregatorBuilder final : public ErisServiceBuilder {
public:
  explicit ErisAggregatorBuilder(void);

  void add_min_clients(uint32_t);
  void add_block_size(size_t);

private:
  friend class ErisClient;

  size_t block_size_;
  uint32_t min_clients_;
};
