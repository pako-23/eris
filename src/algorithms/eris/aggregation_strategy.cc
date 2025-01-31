#include "algorithms/eris/aggregation_strategy.h"
#include "algorithms/eris/aggregator.pb.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <vector>

eris::WeightUpdate WeightedAverage::aggregate(
    uint32_t round, const std::vector<eris::WeightSubmissionRequest> &updates) {
  eris::WeightUpdate update;
  uint32_t samples = 0;

  update.set_round(round);

  update.mutable_weight()->Resize(updates[0].weight_size(), 0.0);
  for (const auto &req : updates) {
    samples += req.samples();

    for (int i = 0; i < req.weight_size(); ++i)
      *update.mutable_weight()->Mutable(i) += req.samples() * req.weight(i);
  }

  for (int i = 0; i < update.weight_size(); ++i)
    *update.mutable_weight()->Mutable(i) = update.weight(i) / samples;

  return update;
}

Soteria::Soteria(float gamma) : reference_{}, prev_{}, gamma_{gamma} {}

void Soteria::configure(const std::vector<float> &fragment) {
  reference_.resize(fragment.size());
  std::copy(fragment.begin(), fragment.end(), std::back_inserter(prev_));
}

eris::WeightUpdate
Soteria::aggregate(uint32_t round,
                   const std::vector<eris::WeightSubmissionRequest> &updates) {
  eris::WeightUpdate update;
  uint32_t samples = std::accumulate(
      updates.begin(), updates.end(), 0,
      [](uint32_t acc, const eris::WeightSubmissionRequest &update) {
        return acc + update.samples();
      });

  std::vector<float> sparse_grads(reference_.size());

  for (const auto &req : updates)
    for (int i = 0; i < req.weight_size(); ++i)
      sparse_grads[i] += (req.weight(i) - prev_[i]) * req.samples();

  for (std::vector<float>::size_type i = 0; i < sparse_grads.size(); ++i) {
    float shifted = sparse_grads[i] / samples;
    sparse_grads[i] = shifted + reference_[i];
    prev_[i] += sparse_grads[i];
    reference_[i] += gamma_ * shifted;
  }

  update.set_round(round);

  update.mutable_weight()->Resize(prev_.size(), 0.0);
  for (int i = 0; i < update.weight_size(); ++i)
    *update.mutable_weight()->Mutable(i) = prev_[i];

  return update;
}
