#include "algorithms/eris/builder.h"
#include "util/networking.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

ErisServiceBuilder::ErisServiceBuilder(void)
    : listen_address_{"0.0.0.0"}, publish_port_{5555}, rpc_port_{5051} {}

void ErisServiceBuilder::add_publish_port(uint16_t port) {
  if (port == 0)
    throw std::invalid_argument{"not a valid port provided"};
  publish_port_ = port;
}

void ErisServiceBuilder::add_rpc_port(uint16_t port) {
  if (port == 0)
    throw std::invalid_argument{"not a valid port provided"};
  rpc_port_ = port;
}

void ErisServiceBuilder::add_listen_address(const std::string &address) {
  if (!valid_ipv4(address))
    throw std::invalid_argument{"not a valid IPv4 address provided"};
  listen_address_ = address;
}

const std::string ErisServiceBuilder::get_rpc_address(void) const {
  return listen_address_ + ":" + std::to_string(rpc_port_);
}

const std::string ErisServiceBuilder::get_zmq_listen_address(void) const {
  return zmq_listen_address(listen_address_, publish_port_);
}

const std::string ErisServiceBuilder::get_zmq_publish_address(void) const {
  return zmq_publish_address(listen_address_, publish_port_);
}

void ErisCoordinatorBuilder::add_rounds(uint32_t rounds) {
  options_.set_rounds(rounds);
}

void ErisCoordinatorBuilder::add_splits(uint32_t splits) {
  options_.set_splits(splits);
}

void ErisCoordinatorBuilder::add_min_clients(uint32_t min_clients) {
  options_.set_min_clients(min_clients);
}

ErisCoordinatorBuilder::ErisCoordinatorBuilder(void)
    : ErisServiceBuilder{}, options_{} {
  options_.set_splits(1);
  options_.set_rounds(1);
  options_.set_min_clients(3);
}

ErisAggregatorBuilder::ErisAggregatorBuilder(void)
    : ErisServiceBuilder{}, min_clients_{4} {}

void ErisAggregatorBuilder::add_min_clients(uint32_t min_clients) {
  min_clients_ = min_clients;
}

void ErisAggregatorBuilder::add_block_size(size_t block_size) {
  block_size_ = block_size;
}
