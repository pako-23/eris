#include "coordinator.h"
#include "client.h"
#include "eris/coordinator.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

template<typename CoordinatorBase = Coordinator>
class PyCoordinator : public CoordinatorBase {
public:
  using CoordinatorBase::CoordinatorBase;

  void run(void) override {
    PYBIND11_OVERRIDE_PURE(void, CoordinatorBase, run,);
  }
};


class PyClient : public Client {
public:
  using Client::Client;
  
  void run(void) override {
    PYBIND11_OVERRIDE_PURE(void, Client, run,);
  }
  
  void get_parameters(void) override {
    PYBIND11_OVERRIDE_PURE(void, Client, get_parameters,);
  }

  void fit(void) override {
    PYBIND11_OVERRIDE_PURE(void, Client, fit,);
  }

  void evaluate(void) override {
    PYBIND11_OVERRIDE_PURE(void, Client, evaluate,);
  }
};

template<typename T>
class PyCoordinatorImpl : public PyCoordinator<T> {
public:
  using PyCoordinator<T>::PyCoordinator;

  void run(void) override {
    PYBIND11_OVERRIDE(void, T, run,);
  }
};


PYBIND11_MODULE(eris, m) {
  py::class_<Coordinator, PyCoordinator<>>(m, "Coordinator");
  py::class_<ErisCoordinator, Coordinator, PyCoordinatorImpl<ErisCoordinator>>(m, "ErisCoordinator");
  py::class_<Client, PyClient>(m, "Client")
    .def(py::init<>())
    .def("run", &Client::run)
    .def("get_parameters", &Client::get_parameters)
    .def("fit", &Client::fit)
    .def("evaluate", &Client::evaluate);
}
