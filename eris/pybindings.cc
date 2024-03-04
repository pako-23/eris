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

  void start(void) override { PYBIND11_OVERRIDE_PURE(void, Coordinator, start, ); }
};

class PyClient : public Client {
 public:
  using Client::Client;

  void start(void) override { PYBIND11_OVERRIDE_PURE(void, Client, start, ); }

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
      .def("start", &Coordinator::start);

  py::class_<ErisCoordinator, Coordinator>(m, "ErisCoordinator")
      .def(py::init<const ErisCoordinatorOptions&, const std::string&, uint16_t, uint16_t>())
      .def("start", &ErisCoordinator::start);

  py::class_<Client, PyClient>(m, "Client")
      .def(py::init<>())
      .def("start", &Client::start)
      .def("get_parameters", &Client::get_parameters)
      .def("fit", &Client::fit)
      .def("evaluate", &Client::evaluate);

  py::class_<ErisClient, Client, PyErisClient>(m, "ErisClient")
      .def(py::init<const std::string&, const std::string&, uint16_t>())
      .def("start", &ErisClient::start)
      .def("get_parameters", &ErisClient::get_parameters)
      .def("fit", &ErisClient::fit)
      .def("evaluate", &ErisClient::evaluate);
}
