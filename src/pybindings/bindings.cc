#include "algorithms/eris/client.h"
#include "algorithms/eris/config.h"
#include "algorithms/eris/coordinator.h"
#include "erisfl/coordinator.h"
#include "util/networking.h"
#include <cmath>
#include <cstdint>
#include <optional>
#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
#include <sys/types.h>

namespace py = pybind11;

class PyCoordinator : public Coordinator {
public:
  using Coordinator::Coordinator;

  void
  start(std::optional<std::promise<void>> started = std::nullopt) override {
    PYBIND11_OVERRIDE_PURE(void, Coordinator, start, started);
  }

  void stop(void) override {
    PYBIND11_OVERRIDE_PURE(void, Coordinator, stop, );
  }

  void start(void) { start(std::nullopt); }
};

class PyClientBase {
public:
  virtual ~PyClientBase(void) = default;
  virtual bool join(void) = 0;
  virtual bool train(void) = 0;
  virtual py::list get_parameters(void) = 0;
  virtual py::list get_split_mask(void) = 0;
  virtual void set_parameters(const py::list &parameters) = 0;
  virtual void fit(void) = 0;
  virtual void evaluate(void) = 0;
};

class PyClient : public PyClientBase {
public:
  using PyClientBase::PyClientBase;

  bool join(void) override {
    PYBIND11_OVERRIDE_PURE(bool, PyClientBase, join, );
  }

  bool train(void) override {
    PYBIND11_OVERRIDE_PURE(bool, PyClientBase, train, );
  }

  py::list get_parameters(void) override {
    PYBIND11_OVERRIDE_PURE(py::list, PyClientBase, get_parameters, );
  }

  py::list get_split_mask(void) override {
    PYBIND11_OVERRIDE_PURE(py::list, PyClientBase, get_split_mask, );
  }

  void set_parameters(const py::list &parameters) override {
    PYBIND11_OVERRIDE_PURE(void, PyClientBase, set_parameters, parameters);
  }

  void fit(void) override { PYBIND11_OVERRIDE_PURE(void, PyClientBase, fit, ); }

  void evaluate(void) override {
    PYBIND11_OVERRIDE_PURE(void, PyClientBase, evaluate, );
  }
};

class PyErisClient : public PyClientBase {
public:
  PyErisClient(const std::string &router_address,
               const std::string &subscribe_address)
      : client_{this, router_address, subscribe_address} {}

  bool train(void) override { return client_.train(); }

  bool join(void) override { return client_.join(); }

  py::list get_parameters(void) override {
    PYBIND11_OVERRIDE_PURE(py::list, PyErisClient, get_parameters, );
  }

  py::list get_split_mask(void) override {
    const std::vector<uint32_t> &mapping = client_.get_splitter().get_mapping();
    py::gil_scoped_acquire acquire;
    py::list list;

    size_t shape = 0;
    size_t i = 0;
    while (i < mapping.size()) {
      size_t items = std::accumulate(client_.shapes_[shape].begin(),
                                     client_.shapes_[shape].end(), 1,
                                     std::multiplies<size_t>());

      py::array_t<uint32_t> array(client_.shapes_[shape]);
      auto layer = array.request();

      for (size_t j = 0; j < items; ++j)
        ((uint32_t *)layer.ptr)[j] = mapping[i++];

      list.append(array);
      ++shape;
    }

    py::gil_scoped_release release;
    return list;
  }

  void set_parameters(const py::list &parameters) override {
    PYBIND11_OVERRIDE_PURE(void, PyErisClient, set_parameters, parameters);
  }

  void fit(void) override { PYBIND11_OVERRIDE_PURE(void, PyErisClient, fit, ); }

  bool set_aggregator_config(const std::string &address, uint16_t submit_port,
                             uint16_t publish_port) {
    return client_.set_aggregator_config(address, submit_port, publish_port);
  }

  void evaluate(void) override {
    PYBIND11_OVERRIDE_PURE(void, PyErisClient, evaluate, );
  }

private:
  class ErisClientImpl : public ErisClient<ZMQSocket> {
  public:
    ErisClientImpl(PyErisClient *client, const std::string &router_address,
                   const std::string &subscribe_address)
        : ErisClient<ZMQSocket>{router_address, subscribe_address},
          client_{client}, shapes_initialized{false}, total_size{0} {}

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
    friend class PyErisClient;

    PyErisClient *client_;

    bool shapes_initialized;
    size_t total_size;
    std::vector<std::vector<ssize_t>> shapes_;
  };

  ErisClientImpl client_;
};

PYBIND11_MODULE(eris, m) {
  m.doc() = "A federated learning framework implementing the eris algorithm";

  py::class_<ErisCoordinatorConfig>(m, "ErisCoordinatorConfig")
      .def(py::init<>())
      .def("set_router_port", &ErisCoordinatorConfig::set_router_port,
           py::arg("port"))
      .def("set_router_address", &ErisCoordinatorConfig::set_router_address,
           py::arg("address"))
      .def("set_publish_port", &ErisCoordinatorConfig::set_publish_port,
           py::arg("port"))
      .def("set_publish_address", &ErisCoordinatorConfig::set_publish_address,
           py::arg("address"))
      .def("set_rounds", &ErisCoordinatorConfig::set_rounds, py::arg("rounds"))
      .def("set_splits", &ErisCoordinatorConfig::set_splits, py::arg("splits"))
      .def("set_min_clients", &ErisCoordinatorConfig::set_min_clients,
           py::arg("min_clients"))
      .def("set_split_seed", &ErisCoordinatorConfig::set_split_seed,
           py::arg("split_seed"));

  py::class_<Coordinator, PyCoordinator>(m, "Coordinator")
      .def(py::init<>())
      .def("start", (void(Coordinator::*)(void)) & PyCoordinator::start)
      .def("stop", &Coordinator::stop);

  py::class_<ErisCoordinator<ZMQSocket>, Coordinator>(m, "ErisCoordinator")
      .def(py::init<const ErisCoordinatorConfig &>())
      .def("start", [](ErisCoordinator<ZMQSocket> &self) { self.start(); })
      .def("stop", &ErisCoordinator<ZMQSocket>::stop);

  py::class_<PyClientBase, PyClient>(m, "Client")
      .def(py::init<>())
      .def("join", &PyClientBase::join)
      .def("train", &PyClientBase::train)
      .def("get_parameters", &PyClientBase::get_parameters)
      .def("get_split_mask", &PyClientBase::get_split_mask)
      .def("set_parameters", &PyClientBase::set_parameters,
           py::arg("parameters"))
      .def("fit", &PyClientBase::fit)
      .def("evaluate", &PyClientBase::evaluate);

  py::class_<PyErisClient, PyClientBase,
             std::unique_ptr<PyErisClient, py::nodelete>>(m, "ErisClient")
      .def(py::init<const std::string &, const std::string &>())
      .def("join", &PyErisClient::join,
           py::call_guard<py::gil_scoped_release>())
      .def("train", &PyErisClient::train,
           py::call_guard<py::gil_scoped_release>())
      .def("get_parameters", &PyErisClient::get_parameters,
           py::call_guard<py::gil_scoped_acquire>())
      .def("get_split_mask", &PyErisClient::get_split_mask,
           py::call_guard<py::gil_scoped_acquire>())
      .def("set_parameters", &PyErisClient::set_parameters,
           py::arg("parameters"))
      .def("fit", &PyErisClient::fit, py::call_guard<py::gil_scoped_release>())
      .def("evaluate", &PyErisClient::evaluate,
           py::call_guard<py::gil_scoped_release>())
      .def("set_aggregator_config", &PyErisClient::set_aggregator_config,
           py::arg("address"), py::arg("submit_port"), py::arg("publish_port"));
}
