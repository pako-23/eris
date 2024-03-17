#pragma once

#include "pybind11/pytypes.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

class Client {
public:
  virtual ~Client(){};
  virtual void start(const std::string &) = 0;
  virtual py::list get_parameters(void) = 0;
  virtual void set_parameters(py::list) = 0;
  virtual void fit(void) = 0;
  virtual void evaluate(void) = 0;
};
