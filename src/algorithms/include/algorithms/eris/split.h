#pragma once

#include "algorithms/eris/aggregator.pb.h"
#include "erisfl/client.h"
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
  std::vector<float>
  reassemble(const std::vector<eris::WeightUpdate> &updates) const noexcept;

  /**
   * It splits the given model weights into a list of weights that can be
   * shared with the aggregators for the given round. The position of the
   * resulting weight represents the identifier of the aggregator with whom the
   * weights should be shared.
   *
   * @param parameters The weights are they are coming from the model.
   * @param round The current training round.
   * @return The list of weights that should be shared with the aggregators.
   */
  std::vector<eris::WeightSubmissionRequest>
  split(const Client::fit_result &parameters, uint32_t round) noexcept;

  /**
   * Returns the mapping from weight to aggregator identifier.
   *
   * @return The mapping from weight to aggregator identifier.
   */
  const std::vector<uint32_t> &get_mapping(void) const noexcept {
    return aggregator_mapping_;
  }

  std::vector<float> get_fragment(const std::vector<float> &parameters,
                                  uint32_t fragment_id) noexcept;

private:
  std::vector<uint32_t>
      aggregator_mapping_; /**< The mapping from parameter position to assigned
                              aggregator */
  uint32_t nsplits_; /**< The number of fragments the spliting should produce */
};
