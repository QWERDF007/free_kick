#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

// module name
PYBIND11_MODULE(free_kick, m) {

    // optional module docstring
    m.doc() = R"pbdoc(
        free_kick Python API reference
        ========================

        This is the Python API reference for the free_kick library.
    )pbdoc";

    m.attr("__version__") = "0.0.1";

    m.def("add", &add, "A function that adds two numbers");
}