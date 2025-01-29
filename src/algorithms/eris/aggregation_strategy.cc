#include "algorithms/eris/aggregation_strategy.h"
#include "algorithms/eris/aggregator.pb.h"
#include <algorithm>
#include <cstdint>
#include <numeric>

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

eris::WeightUpdate
Soteria::aggregate(uint32_t round,
                   const std::vector<eris::WeightSubmissionRequest> &updates) {

  eris::WeightUpdate update;
  uint32_t samples = std::accumulate(
      updates.begin(), updates.end(), 0,
      [](uint32_t acc, const eris::WeightSubmissionRequest &update) {
        return acc + update.samples();
      });

  if (reference_.size() == 0) {
    reference_.resize(updates[0].weight_size());
    prev_.resize(updates[0].weight_size());
  }

  std::vector<float> sparse_grads(reference_.size());

  for (const auto &req : updates)
    for (int i = 0; i < req.weight_size(); ++i)
      sparse_grads[i] += (req.weight(i) - prev_[i]) * req.samples();

  for (std::vector<float>::size_type i = 0; i < sparse_grads.size(); ++i) {
    sparse_grads[i] = (sparse_grads[i] / samples) + reference_[i];
    prev_[i] += sparse_grads[i];
    reference_[i] += gamma_ * sparse_grads[i];
  }

  update.set_round(round);

  update.mutable_weight()->Resize(prev_.size(), 0.0);
  for (int i = 0; i < update.weight_size(); ++i)
    *update.mutable_weight()->Mutable(i) = prev_[i];

  return update;
}
