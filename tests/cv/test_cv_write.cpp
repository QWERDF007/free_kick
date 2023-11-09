#include "../values_test.h"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <string>

// clang-format off

std::vector<std::vector<std::string>> testdata = {
    std::vector<std::string>{"test.png"},
    std::vector<std::string>{"test.jpg"},
    std::vector<std::string>{"test.tiff"},
    std::vector<std::string>{"test.bmp"},
};

FREE_KICK_TEST_SUITE_P(OpenCVImwriteTest, ValueList<std::string&, int>
{
    // ext
    { "test.png",  5 },
    { "test.jpg",  5 },
    { "test.tiff", 2 },
    { "test.bmp",  3 },
});

// clang-format on

TEST_P(OpenCVImwriteTest, ext)
{
    std::string ext  = GetParamValue<0>();
    int         size = GetParamValue<1>();

    std::string img_path = "test." + ext;

    cv::Mat src(size, size, CV_32FC1, cv::Scalar(22.5));
    cv::imwrite(img_path, src);
    cv::Mat dst = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    EXPECT_EQ(src.cols, dst.cols);
    EXPECT_EQ(src.rows, dst.rows);
    EXPECT_EQ(src.type(), dst.type());
    EXPECT_EQ(src.at<float>(0, 0), dst.at<float>(0, 0));
}