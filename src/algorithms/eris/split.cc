#include <algorithms/eris/aggregator.pb.h>
#include <algorithms/eris/split.h>
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
