#pragma once

class SplitStrategy {
public:
  virtual double *split(double *) = 0;
  virtual double *reassemble(double *) = 0;
};
