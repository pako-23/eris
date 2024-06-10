#include "algorithms/eris/split.h"
#include "algorithms/eris/aggregator.pb.h"
#include "pybind11/detail/common.h"
#include "pybind11/numpy.h"
#include "pybind11/pytypes.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

RandomSplit::RandomSplit(void)
    : weights_{}, aggregator_map_{}, nsplits_{}, shapes_{}, reassemble_buf_{} {}

void RandomSplit::setup(const py::list &parameters, uint32_t nsplits,
                        uint32_t seed) {
  weights_.clear();
  weights_.resize(nsplits);
  nsplits_ = nsplits;
  reassemble_buf_.resize(nsplits);
  shapes_.resize(parameters.size());

  size_t total_size{0};
  for (size_t i{0}; i < parameters.size(); ++i) {
    py::detail::unchecked_reference<double, -1L> array{
        py::cast<py::array>(parameters[i]).unchecked<double>()};

    shapes_[i].reserve(array.ndim());
    ssize_t elements{array.shape(0)};
    shapes_[i][0] = array.shape(0);

    for (py::ssize_t j{1}; j < array.ndim(); ++j) {
      shapes_[i][j] = array.shape(j);
      elements *= array.shape(j);
    }

    total_size += elements;
  }

  aggregator_map_.reserve(total_size);

  for (uint32_t i{0}; i < nsplits; ++i) {
    for (size_t j{0}; j < get_block_size(i); ++j)
      aggregator_map_.push_back(i);
  }

  std::mt19937 engine{seed};
  std::shuffle(aggregator_map_.begin(), aggregator_map_.end(), engine);
}

void RandomSplit::split(const py::list &parameters, uint32_t round) {
  for (aggregator::Weight &weight : weights_) {
    weight.clear_weight();
    weight.set_round(round);
  }

  size_t base{0};
  for (size_t i{0}; i < parameters.size(); ++i) {
    auto array{
        py::cast<py::array>(parameters[i]).reshape({-1}).unchecked<double>()};

    for (ssize_t j{0}; j < array.shape(0); ++j) {
      weights_[aggregator_map_[base + j]].weight(array[j]);

      spdlog::info("Sending weight {0} to aggr {1}", aggregator_map_[base + j],
                   array[j]);
    }
    base += array.shape(0);
  }
}

size_t RandomSplit::get_block_size(uint32_t block_id) const {
  size_t block_size{aggregator_map_.size() / nsplits_};

  return block_id < aggregator_map_.size() % nsplits_ ? block_size + 1
                                                      : block_size;
}

void RandomSplit::reassemble(const aggregator::WeightUpdate *update,
                             uint32_t aggregator_id) {
  reassemble_buf_[aggregator_id] = *update;
}

py::list RandomSplit::get_parameters(void) {
  py::list parameters;

  size_t i{0};
  size_t size{std::accumulate(shapes_[i].begin(), shapes_[i].end(), 1ul,
                              std::multiplies<size_t>())};
  std::vector<double> layer(size);
  std::vector<size_t> last_index(reassemble_buf_.size());

  size_t acc{0};
  for (const uint32_t aggregator : aggregator_map_) {
    if (acc == size) {
      py::array_t<double> ret(layer.size(), layer.data());
      ret.reshape(shapes_[i]);
      acc = 0;
      ++i;
      parameters.append(ret);
      size_t size{std::accumulate(shapes_[i].begin(), shapes_[i].end(), 1ul,
                                  std::multiplies<size_t>())};
      layer.resize(size);
    }

    layer[acc] = reassemble_buf_[aggregator].weight()[last_index[aggregator]] /
                 reassemble_buf_[aggregator].contributors();
  }

  py::array_t<double> ret(layer.size(), layer.data());
  ret.reshape(shapes_[i]);
  parameters.append(ret);

  return parameters;
}
