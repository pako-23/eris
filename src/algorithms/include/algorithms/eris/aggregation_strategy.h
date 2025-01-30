#pragma once

#include "algorithms/eris/aggregator.pb.h"
#include <cstdint>
#include <vector>

class AggregationStrategy {
public:
  virtual ~AggregationStrategy(void) = default;

  virtual eris::WeightUpdate
  aggregate(uint32_t round,
            const std::vector<eris::WeightSubmissionRequest> &updates) = 0;

  virtual void configure(const std::vector<float> &fragment) {};
};

class WeightedAverage : public AggregationStrategy {
public:
  explicit WeightedAverage(void) = default;
  ~WeightedAverage(void) = default;

  eris::WeightUpdate
  aggregate(uint32_t round,
            const std::vector<eris::WeightSubmissionRequest> &updates) override;
};

class Soteria : public AggregationStrategy {
public:
  explicit Soteria(float gamma);
  ~Soteria(void) = default;

  eris::WeightUpdate
  aggregate(uint32_t round,
            const std::vector<eris::WeightSubmissionRequest> &updates) override;
  void configure(const std::vector<float> &fragment) override;

private:
  std::vector<float> reference_;
  std::vector<float> prev_;
  float gamma_;
};
