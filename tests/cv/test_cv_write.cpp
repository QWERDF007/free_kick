#include "../values_test.h"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <string>

// clang-format off

FREE_KICK_TEST_SUITE_P(OpenCVImwriteTest, ValueList<std::string&, bool>
{
    // ext
    { "test.png",  false },
    { "test.jpg",  false },
    { "test.tiff", true },
    { "test.bmp",  false },
});

// clang-format on

TEST_P(OpenCVImwriteTest, ext)
{
    std::string ext  = GetParamValue<0>();
    bool        cond = GetParamValue<1>();

    std::string img_path = "test." + ext;

    cv::Mat src(20, 20, CV_32FC1, cv::Scalar(22.5));
    cv::imwrite(img_path, src);
    cv::Mat dst = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    EXPECT_EQ(src.cols == dst.cols, true);
    EXPECT_EQ(src.rows == dst.rows, true);
    EXPECT_EQ(src.type() == dst.type(), cond);
    EXPECT_TRUE(src.depth() == CV_32F);
    EXPECT_EQ(dst.depth() == CV_32F, cond);
    if (dst.depth() == CV_32F)
    {
        EXPECT_FLOAT_EQ(src.at<float>(0, 0), dst.at<float>(0, 0));
    }
}