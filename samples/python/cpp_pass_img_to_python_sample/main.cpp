#include "CrashHandler.h"

#include <opencv2/highgui.hpp>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <filesystem>
#include <iostream>
#include <sstream>
#include <type_traits>

void addSysPath(const std::string &python_home)
{
    namespace fs = std::filesystem;
    // 获取当前工作目录
    fs::path current_path = fs::current_path();
    std::cout << "current_path = " << current_path.generic_string() << std::endl;
    pybind11::object sys = pybind11::module_::import("sys");

    std::stringstream path_to_add;
    path_to_add << current_path.generic_string() << '/' << "py_module";
    sys.attr("path").attr("append")(path_to_add.str());
}

class PythonHelper
{
public:
    /**
     * @brief 检查模板类型是否为 float 或 uint8_t
     * 
     * @tparam T 
     */
    template<typename T>
    using is_float_or_uint8 = std::conditional_t<std::is_same<T, float>::value || std::is_same<T, uint8_t>::value,
                                                 std::true_type, std::false_type>;

    template<typename T>
    static typename std::enable_if_t<is_float_or_uint8<T>::value, pybind11::array_t<T>> toNumpy(const cv::Mat &img)
    {
        const size_t rows = img.rows;
        const size_t cols = img.cols;
        const size_t chs  = img.channels();

        bool single_channel = chs == 1;

        const size_t      item_size = img.elemSize1();
        const std::string format    = pybind11::format_descriptor<T>::format();
        pybind11::ssize_t ndim      = single_channel ? 2 : 3;

        std::cout << __FUNCTION__ << ", line " << __LINE__ << ", rows = " << rows << ", cols = " << cols
                  << ", chs = " << chs << ", format = " << format << std::endl;
        std::cout << __FUNCTION__ << ", line " << __LINE__ << ", elem size = " << img.elemSize1()
                  << ", img.step[0] = " << img.step[0] << ", img.step[1] = " << img.step[1] << std::endl;

        pybind11::array::ShapeContainer shape = single_channel ? pybind11::array::ShapeContainer{rows, cols}
                                                               : pybind11::array::ShapeContainer{rows, cols, chs};
        pybind11::array::ShapeContainer strides
            = single_channel ? pybind11::array::ShapeContainer{img.step[0], item_size}
                             : pybind11::array::ShapeContainer{img.step[0], img.step[1], item_size};

        return pybind11::array_t<T>(pybind11::buffer_info(img.data, item_size, format, ndim, shape, strides));
    }

    template<typename T>
    static cv::Mat fromNumpy(const pybind11::object &obj)
    {
        if (!obj.is_none() && pybind11::isinstance<pybind11::array_t<T>>(obj))
        {
            pybind11::array_t<T>  buf  = obj;
            pybind11::buffer_info info = buf.request();

            int rows = info.shape[0];
            int cols = info.shape[1];
            int chs  = info.shape.size() == 2 ? 1 : info.shape[2];
            int type = info.format == "B" ? CV_8UC(chs) : CV_32FC(chs);

            std::cout << __FUNCTION__ << ", line " << __LINE__ << ", rows = " << rows << ", cols = " << cols
                      << ", chs = " << chs << ", type = " << type << std::endl;
            std::cout << __FUNCTION__ << ", line " << __LINE__ << ", item size = " << info.itemsize
                      << ", format = " << info.format << std::endl;

            return cv::Mat(rows, cols, type, info.ptr);
        }
        return cv::Mat();
    }
};

void CV8UImgTest()
{
    const int h = 6;
    const int w = 5;
    cv::Mat   img(h, w, CV_8UC3, cv::Scalar::all(0));
    cv::Rect  roi(1, 1, 3, 3);
    const int chs = img.channels();
    // 填充图
    for (int r = 0; r < h; r++)
    {
        uchar *row_ptr = img.ptr<uchar>(r);
        int    w_      = w * chs;
        for (int c = 0; c < w_; c += chs)
        {
            for (int i = 0; i < chs; ++i) row_ptr[c + i] = r + i;
        }
    }

    // std::cout << cv::format(img, cv::Formatter::FMT_NUMPY) << std::endl;

    // pybind11::module_ module = pybind11::module_::import("pass_img_test");
    // pybind11::object  func   = module.attr("pass_img_test");
    // std::cout << "full img(CV_8U): " << std::endl;
    // pybind11::object ret     = func(PythonHelper::toNumpy<uint8_t>(img));
    // cv::Mat          ret_img = PythonHelper::fromNumpy<uint8_t>(ret);
    // double           val     = cv::norm(img, ret_img, cv::NORM_INF);
    // std::cout << "norm = " << val << std::endl;
    // std::cout << pybind11::type::of(ret).attr("__name__").cast<std::string>() << std::endl;
    // std::cout << "isinstance<pybind11::array_t<uint8_t>" << pybind11::isinstance<pybind11::array_t<uint8_t>>(ret)
    //           << std::endl;
    // std::cout << ret.is_none() << std::endl;
    // std::cout << std::endl;
    // std::cout << "sub img(CV_8U): " << std::endl;
    // ret     = func(PythonHelper::toNumpy<uint8_t>(img(roi)));
    // ret_img = PythonHelper::fromNumpy<uint8_t>(ret);
    // val     = cv::norm(img(roi), ret_img, cv::NORM_INF);
    // std::cout << "norm = " << val << std::endl;
    // std::cout << pybind11::type::of(ret).attr("__name__").cast<std::string>() << std::endl;
    // std::cout << pybind11::isinstance<pybind11::array_t<uint8_t>>(ret) << std::endl;
    // std::cout << ret.is_none() << std::endl;
    // std::cout << std::endl;
    std::thread t(
        [img]()
        {
            try
            {
                std::cout << __FUNCTION__ << " line " << __LINE__ << std::endl;
                std::cout << "img rows = " << img.rows << ", cols = " << img.cols << std::endl;
                pybind11::gil_scoped_acquire acquire;
                std::cout << "get gil" << std::endl;
                {
                    pybind11::module_ module = pybind11::module_::import("pass_img_test");
                    std::cout << "pass_img_test" << std::endl;
                    pybind11::object obj = module.attr("Yolov8Detection")("F:/models/yolov8/yolov8s.pt", 640, "cuda:0");

                    pybind11::object ret = obj.attr("detect")(PythonHelper::toNumpy<uint8_t>(img));
                }
            }
            catch (const pybind11::error_already_set &e)
            {
                std::cout << "Python error: " << e.what() << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cout << "std::exception: " << e.what();
            }
        });
    t.join();
}

void CV32FImgTest()
{
    const int h = 6;
    const int w = 5;
    cv::Mat   img(h, w, CV_32FC4, cv::Scalar::all(0));
    cv::Rect  roi(1, 1, 3, 3);
    const int chs = img.channels();

    // 填充图
    for (int r = 0; r < h; r++)
    {
        float *row_ptr = img.ptr<float>(r);
        int    w_      = w * chs;
        for (int c = 0; c < w_; c += chs)
        {
            for (int i = 0; i < chs; ++i) row_ptr[c + i] = r + i;
        }
    }

    // std::cout << cv::format(img, cv::Formatter::FMT_NUMPY) << std::endl;

    pybind11::module_ module = pybind11::module_::import("pass_img_test");
    pybind11::object  func   = module.attr("pass_img_test");
    std::cout << "full img(CV_32F): " << std::endl;
    pybind11::object ret     = func(PythonHelper::toNumpy<float>(img));
    cv::Mat          ret_img = PythonHelper::fromNumpy<float>(ret);
    double           val     = cv::norm(img, ret_img, cv::NORM_INF);
    std::cout << "norm = " << val << std::endl;
    std::cout << pybind11::type::of(ret).attr("__name__").cast<std::string>() << std::endl;
    std::cout << "isinstance<pybind11::array_t<float>" << pybind11::isinstance<pybind11::array_t<float>>(ret)
              << std::endl;
    std::cout << ret.is_none() << std::endl;
    std::cout << std::endl;
    std::cout << "sub img(CV_32F): " << std::endl;
    ret     = func(PythonHelper::toNumpy<float>(img(roi)));
    ret_img = PythonHelper::fromNumpy<float>(ret);
    val     = cv::norm(img(roi), ret_img, cv::NORM_INF);
    std::cout << "norm = " << val << std::endl;
    std::cout << std::endl;

    // func(PythonHelper::toNDArray<double>(img(cv::Rect(1, 1, 3, 3)))); // 错误，类型不匹配
}

int main(int argc, char *argv[])
{
    free_kick::common::CrashHandler crash_handler;
    crash_handler.setup();

    const std::string python_home = "D:/Software/anaconda3/envs/AD";
    // 添加环境变量 PATH

    std::string path_env = std::string(std::getenv("PATH"));

    path_env = (python_home + ";" + path_env);
    // path_env = (python_home + "/Library/mingw-w64/bin" + ";" + path_env);
    // path_env = (python_home + "/Library/usr/bin" + ";" + path_env);
    path_env = (python_home + "/Library/bin" + ";" + path_env);
    // path_env = (python_home + "/Scripts" + ";" + path_env);
    // path_env = (python_home + "/bin" + ";" + path_env);
    _putenv_s("PATH", path_env.c_str());
    // std::cout << "PATH = " << std::string(std::getenv("PATH")) << std::endl;
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 11)
    Py_SetPythonHome(Py_DecodeLocale(python_home.c_str(), nullptr));
    pybind11::initialize_interpreter();
    // pybind11::scoped_interpreter guard{};
#else
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    PyConfig_SetBytesString(&config, &config.home, python_home.c_str());
    pybind11::initialize_interpreter(&config);
    // 初始化Python解释器
    pybind11::scoped_interpreter guard{&config};
    PyConfig_Clear(&config);
#endif
    try
    {
        addSysPath(python_home);
        auto gil = new pybind11::gil_scoped_release();
        CV8UImgTest();
        delete gil;
        // CV32FImgTest();
    }
    catch (const pybind11::error_already_set &e)
    {
        std::cout << "Python error: " << e.what() << std::endl;
        auto traceback = pybind11::module::import("traceback").attr("format_exception");
        auto format_exception
            = traceback(e.type(), e.value() ? e.value() : pybind11::none(), e.trace() ? e.trace() : pybind11::none());
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
