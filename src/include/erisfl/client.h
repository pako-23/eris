#pragma once

#include <cstdint>
#include <utility>
#include <vector>

/**
 * The Client class provides an interface of a process that joins the federated
 * training and locally trains its model.
 */
class Client {
public:
  using fit_result = std::pair<std::vector<float>, uint32_t>;

  /**
   * Deletes an instance of a Client object.
   */
  virtual ~Client(void) = default;

  /**
   * Joins the training and performs the whole training process.
   *
   * @return If the training is successful, it returns true; otherwise, it
   * returns false.
   */
  virtual bool train(void) = 0;

  /**
   * Returns the model parameters.
   *
   * @return The model parameters.
   */
  virtual std::vector<float> get_parameters(void) = 0;

  /**
   * Sets the model parameters.
   *
   * @param parameters The model parameters
   */
  virtual void set_parameters(const std::vector<float> &parameters) = 0;

  /**
   * Performs a training round of the model.
   *
   * @return It returns a tuple where the first element consists of
   * the new model parameters, and the second element is the number of
   * samples used for the training.
   */
  virtual fit_result fit(void) = 0;

  /**
   * Evaluates the model performance.
   */
  virtual void evaluate(void) = 0;
};
