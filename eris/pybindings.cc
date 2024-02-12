#include <pybind11/pybind11.h>

#include "client.h"
#include "coordinator.h"
#include "eris/coordinator.h"

namespace py = pybind11;

class PyCoordinator : public Coordinator {
 public:
  using Coordinator::Coordinator;

  void run(void) override { PYBIND11_OVERRIDE_PURE(void, Coordinator, run, ); }
};

class PyClient : public Client {
 public:
  using Client::Client;

  void run(void) override { PYBIND11_OVERRIDE_PURE(void, Client, run, ); }

  void get_parameters(void) override {
    PYBIND11_OVERRIDE_PURE(void, Client, get_parameters, );
  }

  void fit(void) override { PYBIND11_OVERRIDE_PURE(void, Client, fit, ); }

  void evaluate(void) override {
    PYBIND11_OVERRIDE_PURE(void, Client, evaluate, );
  }
};

PYBIND11_MODULE(eris, m) {
  py::class_<Coordinator, PyCoordinator>(m, "Coordinator")
      .def(py::init<>())
      .def("run", &Coordinator::run);

  py::class_<ErisCoordinator, Coordinator>(m, "ErisCoordinator")
      .def(py::init<>())
      .def("run", &ErisCoordinator::run);

  py::class_<Client, PyClient>(m, "Client")
      .def(py::init<>())
      .def("run", &Client::run)
      .def("get_parameters", &Client::get_parameters)
      .def("fit", &Client::fit)
      .def("evaluate", &Client::evaluate);
}
