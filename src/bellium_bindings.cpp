#include "mlp.hpp"

PYBIND11_MODULE(bellium, m)
{
    // MLP Class
    py::class_<MLP>(m, "MLP")
        .def(py::init<int, py::handle, int, double>())
        .def("attributes", &MLP::attributes)
        .def("init", &MLP::init)
        .def("forward_prop", &MLP::forward_prop);
}