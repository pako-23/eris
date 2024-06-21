#include "algorithms/eris/builder.h"
#include "util/networking.h"
#include <cstdint>
#include <limits>
#include <random>
#include <string>

ErisServiceBuilder::ErisServiceBuilder(void)
    : listen_address_{"0.0.0.0"}, rpc_port_{DEFAULT_ERIS_PORT} {}

bool ErisServiceBuilder::add_rpc_port(uint16_t port) {
  rpc_port_ = port;
  return true;
}

bool ErisServiceBuilder::add_listen_address(const std::string &address) {
  if (!valid_ipv4(address))
    return false;
  listen_address_ = address;
  return true;
}

ErisCoordinatorBuilder::ErisCoordinatorBuilder(void)
    : ErisServiceBuilder{}, options_{} {
  options_.set_splits(1);
  options_.set_rounds(1);
  options_.set_min_clients(3);

  std::random_device dev;
  std::mt19937 rng{dev()};
  std::uniform_int_distribution<std::mt19937::result_type> dist{
      0, std::numeric_limits<uint32_t>::max()};

  options_.set_split_seed(dist(rng));
}

bool ErisCoordinatorBuilder::add_rounds(uint32_t rounds) {
  if (rounds == 0)
    return false;

  options_.set_rounds(rounds);
  return true;
}

bool ErisCoordinatorBuilder::add_splits(uint32_t splits) {
  if (splits == 0)
    return false;
  options_.set_splits(splits);
  return true;
}

bool ErisCoordinatorBuilder::add_min_clients(uint32_t min_clients) {
  if (min_clients == 0)
    return false;
  options_.set_min_clients(min_clients);
  return true;
}

bool ErisCoordinatorBuilder::add_split_seed(uint32_t split_seed) {
  options_.set_split_seed(split_seed);
  return true;
}
