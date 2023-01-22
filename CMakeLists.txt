cmake_minimum_required(VERSION 3.10)
project(bellium)

# Add the Eigen library
include_directories(lib/eigen)

# Add the Pybind11 library
add_subdirectory(lib/pybind11)
include_directories(lib/pybind11/include)

# Add the functions pybind module
pybind11_add_module(bellium src/bellium_bindings.cpp src/functions.cpp src/mlp.cpp)

