#include "algorithms/eris/builder.h"
#include "algorithms/eris/client.h"
#include "algorithms/eris/coordinator.h"
#include "erisfl/coordinator.h"
#include <cstddef>
#include <functional>
#include <numeric>
#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <sys/types.h>
#include <vector>

namespace py = pybind11;

class PyCoordinator : public Coordinator {
public:
  using Coordinator::Coordinator;

  void start(void) override {
    PYBIND11_OVERRIDE_PURE(void, Coordinator, start, );
  }

  void stop(void) override {
    PYBIND11_OVERRIDE_PURE(void, Coordinator, stop, );
  }
};

class PyClientBase {
public:
  virtual bool train(void) = 0;
  virtual py::list get_parameters(void) = 0;
  virtual void set_parameters(const py::list &parameters) = 0;
  virtual void fit(void) = 0;
  virtual void evaluate(void){};
};

class PyClient : public PyClientBase {
public:
  using PyClientBase::PyClientBase;

  bool train(void) override {
    PYBIND11_OVERRIDE_PURE(bool, PyClientBase, train, );
  }

  py::list get_parameters(void) override {
    PYBIND11_OVERRIDE_PURE(py::list, PyClientBase, get_parameters, );
  }

  void set_parameters(const py::list &parameters) override {
    PYBIND11_OVERRIDE_PURE(void, PyClientBase, set_parameters, parameters);
  }

  void fit(void) override { PYBIND11_OVERRIDE_PURE(void, PyClientBase, fit, ); }
};

class PyErisClient : public PyClientBase {
public:
  PyErisClient(void) : client_{this} {}

  bool train(void) override { return client_.train(); }

  py::list get_parameters(void) override {
    PYBIND11_OVERRIDE_PURE(py::list, PyErisClient, get_parameters, );
  }

  void set_parameters(const py::list &parameters) override {
    PYBIND11_OVERRIDE_PURE(void, PyErisClient, set_parameters, parameters);
  }

  void fit(void) override { PYBIND11_OVERRIDE_PURE(void, PyErisClient, fit, ); }

  bool set_coordinator_rpc(const std::string &address) {
    return client_.set_coordinator_rpc(address);
  }

  bool set_coordinator_subscription(const std::string &address) {
    return client_.set_coordinator_subscription(address);
  }

  bool set_aggregator_config(const std::string &address, uint16_t submit_port,
                             uint16_t publish_port) {
    return client_.set_aggregator_config(address, submit_port, publish_port);
  }

private:
  class ErisClientImpl : public ErisClient {
  public:
    ErisClientImpl(PyErisClient *client)
        : client_{client}, shapes_initialized{false}, total_size{0} {}

    std::vector<float> get_parameters(void) override {
      py::gil_scoped_acquire acquire;
      py::list list = client_->get_parameters();
      std::vector<float> parameters;

      if (!shapes_initialized) {
        shapes_initialized = true;

        shapes_.resize(list.size());

        for (size_t i = 0; i < list.size(); ++i) {
          auto layer = py::cast<py::array>(list[i]).request();
          total_size += layer.size;
          shapes_[i] = layer.shape;
        }
      }

      parameters.resize(total_size);
      size_t len = 0;

      for (const auto &item : list) {
        auto layer = py::cast<py::array>(item).request();

        for (ssize_t i = 0; i < layer.size; ++i)
          parameters[len++] = ((float *)layer.ptr)[i];
      }
      py::gil_scoped_release release;

      return parameters;
    }

    void set_parameters(const std::vector<float> &parameters) override {
      py::gil_scoped_acquire acquire;
      py::list list;

      size_t shape = 0;
      size_t i = 0;
      while (i < parameters.size()) {
        size_t items =
            std::accumulate(shapes_[shape].begin(), shapes_[shape].end(), 1,
                            std::multiplies<size_t>());

        py::array_t<float> array(shapes_[shape]);
        auto layer = array.request();

        for (size_t j = 0; j < items; ++j)
          ((float *)layer.ptr)[j] = parameters[i++];

        list.append(array);
        ++shape;
      }

      client_->set_parameters(list);
      py::gil_scoped_release release;
    }

    void fit(void) override { client_->fit(); }

    void evaluate(void) override { client_->evaluate(); };

  private:
    PyErisClient *client_;

    bool shapes_initialized;
    size_t total_size;
    std::vector<std::vector<ssize_t>> shapes_;
  };

  ErisClientImpl client_;
};

PYBIND11_MODULE(eris, m) {
  m.doc() = "A federated learning framework implementing the eris algorithm";

  py::class_<Coordinator, PyCoordinator>(m, "Coordinator")
      .def(py::init<>())
      .def("start", &Coordinator::start)
      .def("stop", &Coordinator::stop);

  py::class_<ErisCoordinatorBuilder>(m, "ErisCoordinatorBuilder")
      .def(py::init<>())
      .def("set_rpc_port", &ErisCoordinatorBuilder::add_rpc_port,
           py::arg("port"))
      .def("set_rpc_address", &ErisCoordinatorBuilder::add_rpc_listen_address,
           py::arg("address"))
      .def("set_publish_port", &ErisCoordinatorBuilder::add_publish_port,
           py::arg("port"))
      .def("set_publish_address", &ErisCoordinatorBuilder::add_publish_address,
           py::arg("address"))
      .def("set_rounds", &ErisCoordinatorBuilder::add_rounds, py::arg("rounds"))
      .def("set_splits", &ErisCoordinatorBuilder::add_splits, py::arg("splits"))
      .def("set_min_clients", &ErisCoordinatorBuilder::add_min_clients,
           py::arg("min_clients"))
      .def("set_split_seed", &ErisCoordinatorBuilder::add_split_seed,
           py::arg("split_seed"));

  py::class_<ErisCoordinator, Coordinator>(m, "ErisCoordinator")
      .def(py::init<const ErisCoordinatorBuilder &>())
      .def("start", &ErisCoordinator::start)
      .def("stop", &ErisCoordinator::stop);

  py::class_<PyClientBase, PyClient>(m, "Client")
      .def(py::init<>())
      .def("train", &PyClientBase::train)
      .def("get_parameters", &PyClientBase::get_parameters)
      .def("set_parameters", &PyClientBase::set_parameters,
           py::arg("parameters"))
      .def("fit", &PyClientBase::fit)
      .def("evaluate", &PyClientBase::evaluate);

  py::class_<PyErisClient, PyClientBase,
             std::unique_ptr<PyErisClient, py::nodelete>>(m, "ErisClient")
      .def(py::init<>())
      .def("train", &PyErisClient::train,
           py::call_guard<py::gil_scoped_release>())
      .def("get_parameters", &PyErisClient::get_parameters,
           py::call_guard<py::gil_scoped_acquire>())
      .def("set_parameters", &PyErisClient::set_parameters,
           py::arg("parameters"))
      .def("fit", &PyErisClient::fit, py::call_guard<py::gil_scoped_release>())
      .def("evaluate", &PyErisClient::evaluate,
           py::call_guard<py::gil_scoped_release>())
      .def("set_coordinator_rpc", &PyErisClient::set_coordinator_rpc,
           py::arg("address"))
      .def("set_coordinator_subscription",
           &PyErisClient::set_coordinator_subscription, py::arg("address"))
      .def("set_aggregator_config", &PyErisClient::set_aggregator_config,
           py::arg("address"), py::arg("submit_port"), py::arg("publish_port"));
}
