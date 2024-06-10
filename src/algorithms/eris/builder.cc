#include "algorithms/eris/builder.h"
#include "util/networking.h"
#include <cstdint>
#include <string>

ErisServiceBuilder::ErisServiceBuilder(void)
    : listen_address_{"0.0.0.0"}, rpc_port_{5051} {}

bool ErisServiceBuilder::add_rpc_port(uint16_t port) {
  if (port == 0)
    return false;
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

// ErisAggregatorBuilder::ErisAggregatorBuilder(void)
//     : ErisServiceBuilder{}, min_clients_{4} {}

// void ErisAggregatorBuilder::add_min_clients(uint32_t min_clients) {
//   min_clients_ = min_clients;
// }

// void ErisAggregatorBuilder::add_block_size(size_t block_size) {
//   block_size_ = block_size;
// }
