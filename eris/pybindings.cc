#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>

#include "algorithms/eris/client.h"
#include "algorithms/eris/coordinator.h"
#include "erisfl/client.h"
#include "erisfl/coordinator.h"

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

class PyErisClient : public ErisClient {
 public:
  using ErisClient::ErisClient;

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

  py::class_<ErisClient, Client, PyErisClient>(m, "ErisClient")
      .def(py::init<const std::string&, const std::string&, uint16_t>())
      .def("run", &ErisClient::run)
      .def("get_parameters", &ErisClient::get_parameters)
      .def("fit", &ErisClient::fit)
      .def("evaluate", &ErisClient::evaluate);
}
