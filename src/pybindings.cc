#include "algorithms/eris/builder.h"
#include "algorithms/eris/client.h"
#include "algorithms/eris/coordinator.h"
#include "erisfl/client.h"
#include "erisfl/coordinator.h"
#include <memory>
#include <optional>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

class PyCoordinator : public Coordinator {
public:
  using Coordinator::Coordinator;

  void start(void) override {
    PYBIND11_OVERRIDE_PURE(void, Coordinator, start, );
  }
};

class PyClient : public Client {
public:
  using Client::Client;

  void start(const std::string &coordinator_address) override {
    PYBIND11_OVERRIDE_PURE(void, Client, start, coordinator_address);
  }

  py::list get_parameters(void) override {
    PYBIND11_OVERRIDE_PURE(py::list, Client, get_parameters, );
  }

  void set_parameters(py::list parameters) override {
    PYBIND11_OVERRIDE_PURE(void, Client, set_parameters, parameters);
  }

  void fit(void) override { PYBIND11_OVERRIDE_PURE(void, Client, fit, ); }

  void evaluate(void) override {
    PYBIND11_OVERRIDE_PURE(void, Client, evaluate, );
  }
};

// class PyErisClient : public ErisClient {
// public:
//   using ErisClient::ErisClient;

//   py::list get_parameters(void) override {
//     PYBIND11_OVERRIDE_PURE(py::list, ErisClient, get_parameters, );
//   }

//   void set_parameters(py::list parameters) override {
//     PYBIND11_OVERRIDE_PURE(void, Client, set_parameters, parameters);
//   }

//   void fit(void) override { PYBIND11_OVERRIDE_PURE(void, ErisClient, fit, );
//   }

//   void evaluate(void) override {
//     PYBIND11_OVERRIDE_PURE(void, ErisClient, evaluate, );
//   }
// };

PYBIND11_MODULE(eris, m) {
  py::class_<ErisCoordinatorBuilder>(m, "ErisCoordinatorBuilder")
      .def(py::init<>())
      .def("add_rpc_port", &ErisCoordinatorBuilder::add_rpc_port,
           py::arg("port"))
      .def("add_listen_address", &ErisCoordinatorBuilder::add_listen_address,
           py::arg("address"))
      .def("add_rounds", &ErisCoordinatorBuilder::add_rounds, py::arg("rounds"))
      .def("add_splits", &ErisCoordinatorBuilder::add_splits, py::arg("splits"))
      .def("add_min_clients", &ErisCoordinatorBuilder::add_min_clients,
           py::arg("min_clients"));

  py::class_<Coordinator, PyCoordinator, std::shared_ptr<Coordinator>>(
      m, "Coordinator")
      .def(py::init<>())
      .def("start", &Coordinator::start);

  py::class_<ErisCoordinator, Coordinator, std::shared_ptr<ErisCoordinator>>(
      m, "ErisCoordinator")
      .def(py::init([](const ErisCoordinatorBuilder &builder) {
        return std::unique_ptr<ErisCoordinator>(new ErisCoordinator(builder));
      }))
      .def("start", &ErisCoordinator::start);

  // py::class_<ErisAggregatorBuilder>(m, "ErisAggregatorBuilder")
  //     .def(py::init<>())

  //     .def("add_rpc_port", &ErisAggregatorBuilder::add_rpc_port,
  //          py::arg("port"))
  //     .def("add_listen_address", &ErisAggregatorBuilder::add_listen_address,
  //          py::arg("address"));

  // py::class_<Client, PyClient, std::shared_ptr<Client>>(m, "Client")
  //     .def(py::init<>())
  //     .def("start", &Client::start)
  //     .def("get_parameters", &Client::get_parameters)
  //     .def("set_parameters", &Client::set_parameters)
  //     .def("fit", &Client::fit)
  //     .def("evaluate", &Client::evaluate);

  // py::class_<ErisClient, Client, PyErisClient, std::shared_ptr<ErisClient>>(
  //     m, "ErisClient")
  //     .def(py::init<std::optional<ErisAggregatorBuilder>>(),
  //          py::arg("aggregator_builder") = std::nullopt)
  //     .def("start", &ErisClient::start)
  //     .def("get_parameters", &ErisClient::get_parameters)
  //     .def("set_parameters", &ErisClient::set_parameters)
  //     .def("fit", &ErisClient::fit)
  //     .def("evaluate", &ErisClient::evaluate);
}
