#pragma once

#include <vector>

class Client {
public:
  virtual ~Client(void){};
  virtual void start(void) = 0;
  virtual std::vector<double> get_parameters(void) = 0;
  virtual void set_parameters(const std::vector<double> &parameters) = 0;
  virtual void fit(void) = 0;
  virtual void evaluate(void) = 0;
};
