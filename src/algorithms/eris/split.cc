#include "algorithms/eris/split.h"
#include "algorithms/eris/aggregator.pb.h"
#include <cstddef>
#include <cstdint>
#include <random>

void RandomSplit::configure(size_t parameters, uint32_t splits,
                            uint32_t seed) noexcept {
  aggregator_mapping_.reserve(parameters);
  nsplits_ = splits;

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

std::vector<eris::WeightSubmissionRequest>
RandomSplit::split(const Client::fit_result &parameters,
                   uint32_t round) noexcept {
  std::vector<eris::WeightSubmissionRequest> fragments;
  fragments.resize(nsplits_);

  for (uint32_t i = 0; i < fragments.size(); ++i) {
    fragments[i].set_round(round);
    fragments[i].set_samples(parameters.second);
  }

  for (size_t i = 0; i < parameters.first.size(); ++i)
    fragments[aggregator_mapping_[i]].add_weight(parameters.first[i]);

  return fragments;
}

std::vector<float> RandomSplit::reassemble(
    const std::vector<eris::WeightUpdate> &updates) const noexcept {
  std::vector<float> parameters(aggregator_mapping_.size());
  std::vector<int> assigned(updates.size(), 0);

  for (size_t i = 0; i < aggregator_mapping_.size(); ++i) {
    uint32_t fragment_id = aggregator_mapping_[i];
    const eris::WeightUpdate &update = updates[fragment_id];
    parameters[i] = update.weight(assigned[fragment_id]);
    ++assigned[fragment_id];
  }

  return parameters;
}
