#include "coordinator.h"
#include "client.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename T = Coordinator>
class PyCoordinator : public T {
public:
  using T::T;

  void run(void) override
  {
    PYBIND11_OVERRIDE_PURE(void, T, run,);
  }
};

template <typename T>
class PyCoordinatorImpl : public PyCoordinator<T> {
public:
  using PyCoordinator<T>::PyCoordinator;

  void run(void) override
  {
    PYBIND11_OVERRIDE(void, T, run,);
  }
};


PYBIND11_MODULE(eris, m) {
  py::class_<Coordinator, PyCoordinator<>>(m, "Coordinator");
  py::class_<ErisCoordinator, Coordinator, PyCoordinatorImpl<ErisCoordinator>>(m, "ErisCoordinator");
}
