#include "algorithms/eris/builder.h"
#include "algorithms/eris/client.h"
#include "util/networking.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>

ErisServiceBuilder::ErisServiceBuilder(void) noexcept
    : rpc_listen_address_{"0.0.0.0"}, rpc_port_{DEFAULT_ERIS_RPC_PORT},
      publish_address_{"0.0.0.0"}, publish_port_{DEFAULT_ERIS_PUBSUB_PORT} {}

bool ErisServiceBuilder::add_rpc_port(uint16_t port) noexcept {
  rpc_port_ = port;
  return true;
}

bool ErisServiceBuilder::add_rpc_listen_address(
    const std::string &address) noexcept {
  if (!valid_ipv4(address))
    return false;
  rpc_listen_address_ = address;
  return true;
}

bool ErisServiceBuilder::add_publish_address(
    const std::string &address) noexcept {
  if (!valid_ipv4(address))
    return false;
  publish_address_ = address;
  return true;
}

bool ErisServiceBuilder::add_publish_port(uint16_t port) noexcept {
  publish_port_ = port;
  return true;
}

ErisCoordinatorBuilder::ErisCoordinatorBuilder(void) noexcept
    : ErisServiceBuilder{}, options_{} {
  options_.set_splits(DEFAULT_ERIS_SPLITS);
  options_.set_rounds(DEFAULT_ERIS_ROUNDS);
  options_.set_min_clients(DEFAULT_ERIS_MIN_CLIENTS);

  std::random_device dev;
  std::mt19937 rng{dev()};
  std::uniform_int_distribution<std::mt19937::result_type> dist{
      0, std::numeric_limits<uint32_t>::max()};

  options_.set_split_seed(dist(rng));
}

bool ErisCoordinatorBuilder::add_rounds(uint32_t rounds) noexcept {
  if (rounds == 0)
    return false;

  options_.set_rounds(rounds);
  return true;
}

bool ErisCoordinatorBuilder::add_splits(uint32_t splits) noexcept {
  if (splits == 0)
    return false;
  options_.set_splits(splits);
  return true;
}

bool ErisCoordinatorBuilder::add_min_clients(uint32_t min_clients) noexcept {
  if (min_clients == 0)
    return false;
  options_.set_min_clients(min_clients);
  return true;
}

bool ErisCoordinatorBuilder::add_split_seed(uint32_t split_seed) noexcept {
  options_.set_split_seed(split_seed);
  return true;
}

ErisAggregatorBuilder::ErisAggregatorBuilder(uint32_t fragment_id,
                                             size_t fragment_size)
    : fragment_id_{fragment_id}, fragment_size_{fragment_size},
      min_clients_{DEFAULT_ERIS_MIN_CLIENTS} {
  if (fragment_size_ == 0)
    throw std::invalid_argument{"A fragment cannot have size 0"};
}

bool ErisAggregatorBuilder::add_min_clients(uint32_t min_clients) noexcept {
  if (min_clients == 0)
    return false;
  min_clients_ = min_clients;
  return true;
}
