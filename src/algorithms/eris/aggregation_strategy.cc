#include "algorithms/eris/aggregation_strategy.h"
#include "algorithms/eris/aggregator.pb.h"
#include <cstdint>

eris::WeightUpdate WeightedAverage::aggregate(
    uint32_t round, const std::vector<eris::WeightSubmissionRequest> &updates) {
  eris::WeightUpdate update;
  uint32_t denom = 0;

  update.set_round(round);

  update.mutable_weight()->Resize(updates[0].weight_size(), 0.0);
  for (const auto &req : updates) {
    denom += req.samples();

    for (int i = 0; i < req.weight_size(); ++i)
      *update.mutable_weight()->Mutable(i) += req.samples() * req.weight(i);
  }

  for (int i = 0; i < update.weight_size(); ++i)
    *update.mutable_weight()->Mutable(i) = update.weight(i) / denom;

  return update;
}
