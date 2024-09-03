#include "CrashHandler.h"

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace py = pybind11;

void test() {}

void setupPythonHome()
{
    char  *pValue;
    size_t len;
    _putenv("PYTHONHOME=D:/Software/anaconda3/envs/py38/");
    _dupenv_s(&pValue, &len, "PYTHONHOME");
    printf("PYTHONHOME = %s\n", pValue);
    free(pValue);
}

void PythonStandardTest()
{
    std::cout << "Hello World!" << std::endl;
    Py_SetPythonHome(L"D:/Software/anaconda3/envs/py38/");
    Py_Initialize();
    std::cout << "Python initialized" << std::endl;

    int ret = PyRun_SimpleString("import sys; print(sys.version); print(sys.path);");
    std::cout << "ret = " << ret << std::endl;
    ret = PyRun_SimpleString("import cv2; print(cv2.__version__);");
    std::cout << "ret = " << ret << std::endl;
    std::cout << "Python finalized" << std::endl;
    Py_Finalize();
}

void Pybind11Test()
{
    namespace py = pybind11;
    std::cout << "Hello World!" << std::endl;
    Py_SetPythonHome(L"D:/Software/anaconda3/envs/test2");
    py::scoped_interpreter guard{};
    std::cout << "import module sys" << std::endl;
    py::exec("import sys; print(sys.version); print(sys.path);");
    std::cout << "import module cv2" << std::endl;
    py::exec("import cv2; print(cv2.__version__);");
}

void InitSysPath()
{
    namespace fs = std::filesystem;
    // 获取当前工作目录
    fs::path current_path = fs::current_path();
    std::cout << "current_path = " << current_path.generic_string() << std::endl;
    py::object sys          = py::module_::import("sys");
    py::object version_info = sys.attr("version_info");
    int        major        = version_info.attr("major").cast<int>();
    int        minor        = version_info.attr("minor").cast<int>();
    int        micro        = version_info.attr("micro").cast<int>();
    std::cout << "Python version: " << major << "." << minor << "." << micro << std::endl;
    std::stringstream path_to_add;
    path_to_add << current_path.generic_string() << '/' << "py_module";
    sys.attr("path").attr("append")(path_to_add.str());
}

void PythonCallByString()
{
    std::cout << __FUNCTION__ << " called" << std::endl;
    py::exec("import sys; print(sys.version); print(sys.path);");
}

void PythonClassObjectInstanceAndCall()
{
    std::cout << __FUNCTION__ << " called" << std::endl;
    py::object mymodule = py::module_::import("mymodule");
    py::object MYClass  = mymodule.attr("MyClass");
    py::object my_obj   = MYClass(2);
    py::object res      = my_obj();
    std::cout << "res = " << res.cast<int>() << std::endl;
    res = my_obj();
    std::cout << "res = " << res.cast<int>() << std::endl;
}

int main(int argc, char *argv[])
{
    // free_kick::common::CrashHandler crash_handler;
    // crash_handler.setup();
    // 设置环境变量 PYTHONHOME
    // Py_SetPythonHome(L"D:/Software/anaconda3/envs/test2");
    // py::scoped_interpreter guard{};
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    PyConfig_SetBytesString(&config, &config.home, "H:/Software/Anaconda3/envs/ad");
    // 初始化Python解释器
    py::scoped_interpreter guard{&config};
    PyConfig_Clear(&config);
    try
    {
        InitSysPath();
        PythonCallByString();
        PythonClassObjectInstanceAndCall();
    }
    catch (const py::error_already_set &e)
    {
        std::cout << "Python error: " << e.what() << std::endl;
        auto traceback = py::module::import("traceback").attr("format_exception");
        auto format_exception
            = traceback(e.type(), e.value() ? e.value() : py::none(), e.trace() ? e.trace() : py::none());
        std::stringstream ss;
        for (auto line : format_exception)
        {
            ss << line.cast<std::string>();
        }
        std::cout << ss.str() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "std::exception: " << e.what();
    }
    catch (...)
    {
        std::cout << "Unknown exception";
    }
    return 0;
}