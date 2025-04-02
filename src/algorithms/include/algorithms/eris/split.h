#pragma once

#include <algorithms/eris/aggregator.pb.h>
#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * The RandomSplit is a strategy to fragment the model weights in a
 * randomic-like way.
 */
class RandomSplit final {
public:
  /**
   * It configures a RandomSplit object such that it splits a model with weights
   * of the shape of parameters into splits fragments using seed as seed of
   * the random splitting function.
   *
   * @param parameters The number of weights in the model.
   * @param splits The number of fragments that should be produced by the
   * splitting.
   * @param seed The seed of the randomic splitting function.
   */
  void configure(size_t parameters, uint32_t splits, uint32_t seed) noexcept;

  /**
   * It returns the size of a fragment with a given identifier.
   *
   * @param fragment_id The identifier of the fragment.
   * @return The size of the fragment with identifier equal to fragment_id.
   */
  size_t get_fragment_size(uint32_t fragment_id) const noexcept;

  /**
   * It reassembles the model weights updates coming from the aggregators into a
   * newer version of the model parameters.
   *
   * @param updates The list of weight updates as they are provided by the
   * aggregators.
   * @return The new weights of the model.
   */
  template <class It>
  void
  reassemble(It begin, It end,
             const std::vector<eris::WeightUpdate> &updates) const noexcept {
    std::vector<int> assigned(updates.size(), 0);

    for (size_t i = 0; i < aggregator_mapping_.size() && begin != end;
         ++i, ++begin) {
      uint32_t fragment_id = aggregator_mapping_[i];
      const eris::WeightUpdate &update = updates[fragment_id];
      *begin = update.weight(assigned[fragment_id]);
      ++assigned[fragment_id];
    }
  }

  /**
   * It splits the given model weights into a list of weights that can be
   * shared with the aggregators for the given round. The position of the
   * resulting weight represents the identifier of the aggregator with whom the
   * weights should be shared.
   *
   * @param begin An iterator to delimiting the beginning of a sequence of
   * parameters.
   * @param end An iterator to delimiting the end of a sequence of
   * parameters.
   * @param samples The number of samples used to fit.
   * @param round The current training round.
   * @return The list of weights that should be shared with the aggregators.
   */
  template <class It>
  std::vector<eris::WeightSubmissionRequest>
  split(It begin, It end, uint32_t samples, uint32_t round) noexcept {
    std::vector<eris::WeightSubmissionRequest> fragments;
    fragments.resize(nsplits_);

    for (uint32_t i = 0; i < fragments.size(); ++i) {
      fragments[i].set_round(round);
      fragments[i].set_samples(samples);
    }

    for (size_t i = 0; begin != end; ++begin, ++i)
      fragments[aggregator_mapping_[i]].add_weight(*begin);

    return fragments;
  }

  /**
   * Returns the mapping from weight to aggregator identifier.
   *
   * @return The mapping from weight to aggregator identifier.
   */
  const std::vector<uint32_t> &get_mapping(void) const noexcept {
    return aggregator_mapping_;
  }

  template <class Parameters>
  std::vector<float> get_fragment(const Parameters &parameters,
                                  uint32_t fragment_id) noexcept {
    std::vector<float> fragment(get_fragment_size(fragment_id));
    size_t i = 0;

    auto it = parameters.begin();
    for (std::vector<uint32_t>::size_type j = 0;
         j < parameters.size() && i < fragment.size(); ++j, ++it)
      if (aggregator_mapping_[j] == fragment_id)
        fragment[i++] = *it;

    return fragment;
  }

private:
  std::vector<uint32_t>
      aggregator_mapping_; /**< The mapping from parameter position to assigned
                              aggregator */
  uint32_t nsplits_; /**< The number of fragments the spliting should produce */
};
