#pragma once

#include <vector>

/**
 * The Client class provides an interface of a process that joins the federated
 * training and locally trains its model.
 */
class Client {
public:
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
   */
  virtual void fit(void) = 0;

  /**
   * Evaluates the model performance.
   */
  virtual void evaluate(void) = 0;
};
