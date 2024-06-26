#include "algorithms/eris/split.h"
#include "algorithms/eris/coordinator.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>

RandomSplit::RandomSplit(const std::vector<double> &parameters, uint32_t splits,
                         uint32_t seed) noexcept
    : aggregator_mapping_{}, nsplits_{splits} {
  aggregator_mapping_.reserve(parameters.size());

  for (uint32_t i = 0; i < nsplits_; ++i)
    for (size_t j = 0; j < get_fragment_size(i); ++j)
      aggregator_mapping_.push_back(i);

  std::mt19937 eng{seed};

  for (int i = 0; i < 3; ++i)
    std::shuffle(aggregator_mapping_.begin(), aggregator_mapping_.end(), eng);
}

size_t RandomSplit::get_fragment_size(uint32_t fragment_id) const noexcept {
  if (fragment_id < aggregator_mapping_.capacity() % nsplits_)
    return aggregator_mapping_.capacity() / nsplits_ + 1;
  return aggregator_mapping_.capacity() / nsplits_;
}

std::vector<FragmentWeights>
RandomSplit::split(const std::vector<double> &parameters,
                   uint32_t round) noexcept {
  std::vector<FragmentWeights> fragments;
  fragments.resize(nsplits_);

  for (uint32_t i = 0; i < fragments.size(); ++i)
    fragments[i].set_round(round);

  for (size_t i = 0; i < parameters.size(); ++i)
    fragments[aggregator_mapping_[i]].add_weight(parameters[i]);

  return fragments;
}

std::vector<double> RandomSplit::reassemble(
    const std::vector<WeightUpdate> &updates) const noexcept {
  std::vector<double> parameters(aggregator_mapping_.size());
  std::vector<int> assigned(updates.size(), 0);

  for (size_t i = 0; i < aggregator_mapping_.size(); ++i) {
    uint32_t fragment_id = aggregator_mapping_[i];
    const WeightUpdate &update = updates[fragment_id];
    parameters[i] =
        update.weight(assigned[fragment_id]) / update.contributors();
    ++assigned[fragment_id];
  }

  return parameters;
}
