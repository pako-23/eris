#pragma once

#include "algorithms/eris/aggregator.pb.h"

#include <cstddef>
#include <cstdint>
#include <pybind11/pytypes.h>
#include <vector>

namespace py = pybind11;

class RandomSplit final {
public:
  RandomSplit(void);

  void setup(const py::list &, uint32_t, uint32_t);
  void split(const py::list &, uint32_t);
  size_t get_block_size(uint32_t) const;

  inline size_t size() const { return weights_.size(); }
  inline const aggregator::Weight &operator[](int i) const {
    return weights_[i];
  }

  void reassemble(const aggregator::WeightUpdate *, uint32_t);
  py::list get_parameters(void);

private:
  std::vector<aggregator::Weight> weights_;
  std::vector<uint32_t> aggregator_map_;
  uint32_t nsplits_;

  std::vector<std::vector<size_t>> shapes_;
  std::vector<aggregator::WeightUpdate> reassemble_buf_;
};
