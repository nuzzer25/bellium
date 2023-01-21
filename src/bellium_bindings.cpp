#include "neural_network.hpp"

PYBIND11_MODULE(bellium, m)
{
    // Creating Model
    m.def("create", &create, py::return_value_policy::reference, "Creates an Instance of a Neural Network Model");

    // Model Class
    py::class_<Model>(m, "Model")
        .def(py::init<std::string, std::vector<int>, double>())
        .def("attributes", &Model::attributes, "Views Model Attributes");
}