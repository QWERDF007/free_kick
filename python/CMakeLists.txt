set(PYBIND11_PYTHON_VERSION 3.7 CACHE STRING "")
# find_package(Python 3.7 COMPONENTS Development Interpreter)
add_subdirectory(${PYBIND11_SOURCE_DIR} pybind11)


# pybind11 method:
pybind11_add_module(example example.cpp)

set_target_properties(example PROPERTIES
    DEBUG_POSTFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)