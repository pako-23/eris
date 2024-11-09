#include "algorithms/eris/config.h"
#include <cstdint>
#include <random>
#include <stdexcept>

ErisCoordinatorConfig::ErisCoordinatorConfig(void) noexcept : options_{} {
  options_.set_splits(DEFAULT_ERIS_SPLITS);
  options_.set_rounds(DEFAULT_ERIS_ROUNDS);
  options_.set_min_clients(DEFAULT_ERIS_MIN_CLIENTS);
  set_publish_port(DEFAULT_ERIS_PUBLISH_PORT);
  set_router_port(DEFAULT_ERIS_ROUTER_PORT);

  std::random_device dev;
  std::mt19937 rng{dev()};
  std::uniform_int_distribution<std::mt19937::result_type> dist{
      0, std::numeric_limits<uint32_t>::max()};

  options_.set_split_seed(dist(rng));
}

bool ErisCoordinatorConfig::set_rounds(uint32_t rounds) noexcept {
  if (rounds == 0)
    return false;

  options_.set_rounds(rounds);
  return true;
}

bool ErisCoordinatorConfig::set_splits(uint32_t splits) noexcept {
  if (splits == 0)
    return false;
  options_.set_splits(splits);
  return true;
}

bool ErisCoordinatorConfig::set_min_clients(uint32_t min_clients) noexcept {
  if (min_clients == 0)
    return false;
  options_.set_min_clients(min_clients);
  return true;
}

ErisAggregatorConfig::ErisAggregatorConfig(uint32_t fragment_id,
                                           size_t fragment_size)
    : fragment_id_{fragment_id}, fragment_size_{fragment_size},
      min_clients_{DEFAULT_ERIS_MIN_CLIENTS} {
  set_publish_port(DEFAULT_ERIS_PUBLISH_PORT);
  set_router_port(DEFAULT_ERIS_ROUTER_PORT);
  if (fragment_size_ == 0)
    throw std::invalid_argument{"A fragment cannot have size 0"};
}

bool ErisAggregatorConfig::set_min_clients(uint32_t min_clients) noexcept {
  if (min_clients == 0)
    return false;
  min_clients_ = min_clients;
  return true;
}
