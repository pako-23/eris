#pragma once

#include <vector>

class Client {
public:
  virtual ~Client(void) = default;
  virtual bool train(void) = 0;
  virtual std::vector<float> get_parameters(void) = 0;
  virtual void set_parameters(const std::vector<float> &parameters) = 0;
  virtual void fit(void) = 0;
  virtual void evaluate(void){};
};
