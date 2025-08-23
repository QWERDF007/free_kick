#include "common/utility.h"

#include "morphology_unified.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace free_kick::cuda::ops;

// Google Test 参数化结构
struct MorphTestParams
{
    int kernel_size; // kernel size
};

class MorphologyCudaTest : public ::testing::TestWithParam<MorphTestParams>
{
protected:
    void SetUp() override
    {
        std::string path = "D:/Project/dianjiao/2025_08_18/picture_1_2025_08_18_18_02_25_059.png";
        test_image_      = cv::imread(path, cv::IMREAD_GRAYSCALE);
        // // 创建一个测试图像（4096x2048，包含各种几何形状）
        // test_image_ = cv::Mat::zeros(4096, 2048, CV_8UC1);

        // // 添加一些几何形状用于测试
        // // 矩形
        // for (int i = 0; i < 50; ++i)
        // {
        //     int x = rand() % test_image_.cols;
        //     int y = rand() % test_image_.rows;
        //     int w = rand() % 200;
        //     int h = rand() % 200;
        //     cv::rectangle(test_image_, cv::Rect(x, y, w, h), cv::Scalar(255), -1);
        //     x     = rand() % test_image_.cols;
        //     y     = rand() % test_image_.rows;
        //     int r = rand() % 50;
        //     cv::circle(test_image_, cv::Point(x, y), r, cv::Scalar(255), -1);
        // }

        // // 椭圆
        // cv::ellipse(test_image_, cv::Point(200, 250), cv::Size(60, 30), 45, 0, 360, cv::Scalar(255), -1);
        // cv::ellipse(test_image_, cv::Point(700, 400), cv::Size(80, 40), 0, 0, 360, cv::Scalar(255), -1);
        // cv::ellipse(test_image_, cv::Point(400, 500), cv::Size(45, 70), 90, 0, 360, cv::Scalar(255), -1);

        // // 三角形
        // std::vector<cv::Point> triangle1 = {cv::Point(1000, 100), cv::Point(1100, 200), cv::Point(900, 200)};
        // cv::fillPoly(test_image_, triangle1, cv::Scalar(255));
        // std::vector<cv::Point> triangle2 = {cv::Point(1200, 300), cv::Point(1300, 400), cv::Point(1100, 400)};
        // cv::fillPoly(test_image_, triangle2, cv::Scalar(255));

        // // 多边形
        // std::vector<cv::Point> hexagon = {cv::Point(1400, 150), cv::Point(1450, 130), cv::Point(1500, 150),
        //                                   cv::Point(1500, 200), cv::Point(1450, 220), cv::Point(1400, 200)};
        // cv::fillPoly(test_image_, hexagon, cv::Scalar(255));

        // // 线条
        // cv::line(test_image_, cv::Point(100, 600), cv::Point(300, 650), cv::Scalar(255), 5);
        // cv::line(test_image_, cv::Point(400, 600), cv::Point(600, 600), cv::Scalar(255), 8);
        // cv::line(test_image_, cv::Point(700, 580), cv::Point(900, 620), cv::Scalar(255), 3);

        // // 十字形
        // cv::line(test_image_, cv::Point(1600, 200), cv::Point(1700, 200), cv::Scalar(255), 10);
        // cv::line(test_image_, cv::Point(1650, 150), cv::Point(1650, 250), cv::Scalar(255), 10);

        // // 星形
        // std::vector<cv::Point> star
        //     = {cv::Point(1800, 100), cv::Point(1820, 140), cv::Point(1860, 140), cv::Point(1830, 170),
        //        cv::Point(1840, 210), cv::Point(1800, 190), cv::Point(1760, 210), cv::Point(1770, 170),
        //        cv::Point(1740, 140), cv::Point(1780, 140)};
        // cv::fillPoly(test_image_, star, cv::Scalar(255));
        // // 一些噪点
        // for (int i = 0; i < 1000; ++i)
        // {
        //     int x = rand() % test_image_.cols;
        //     int y = rand() % test_image_.rows;
        //     if (rand() % 2)
        //     {
        //         cv::circle(test_image_, cv::Point(x, y), 3, cv::Scalar(255), -1);
        //     }
        // }

        img_w_      = test_image_.cols;
        img_h_      = test_image_.rows;
        img_stride_ = static_cast<int>(test_image_.step[0]);

        // 分配 CUDA 内存
        size_t img_bytes = img_h_ * img_stride_;
        CUDA_CHECK(cudaMalloc(&d_input_, img_bytes));
        CUDA_CHECK(cudaMalloc(&d_output_, img_bytes));
        CUDA_CHECK(cudaMalloc(&d_tmp1_, img_bytes));
        CUDA_CHECK(cudaMalloc(&d_tmp2_, img_bytes));

        // 复制图像到 GPU
        CUDA_CHECK(cudaMemcpy(d_input_, test_image_.data, img_bytes, cudaMemcpyHostToDevice));

        // 创建 CUDA stream
        CUDA_CHECK(cudaStreamCreate(&stream_));

        // 设置默认 block 维度
        block_dim_ = dim3(32, 16);
    }

    void TearDown() override
    {
        if (d_input_)
            cudaFree(d_input_);
        if (d_output_)
            cudaFree(d_output_);
        if (d_tmp1_)
            cudaFree(d_tmp1_);
        if (d_tmp2_)
            cudaFree(d_tmp2_);
        if (d_se_v1_)
            cudaFree(d_se_v1_);
        if (d_se_v2_)
            cudaFree(d_se_v2_);
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    // 准备结构元素（为 v1 和 v2 版本）
    void prepareStructuringElement(const cv::Mat &kernel)
    {
        se_w_     = kernel.cols;
        se_h_     = kernel.rows;
        anchor_x_ = se_w_ / 2;
        anchor_y_ = se_h_ / 2;

        // 为 v1 准备（直接复制掩码）
        if (d_se_v1_)
            cudaFree(d_se_v1_);
        size_t se_bytes = se_w_ * se_h_ * sizeof(uint8_t);
        CUDA_CHECK(cudaMalloc(&d_se_v1_, se_bytes));
        CUDA_CHECK(cudaMemcpy(d_se_v1_, kernel.data, se_bytes, cudaMemcpyHostToDevice));

        // 为 v2 准备（构建偏移列表）
        auto offsets = buildOffsetList(kernel.data, se_w_, se_h_, anchor_x_, anchor_y_);
        n_offsets_   = static_cast<int>(offsets.size());

        if (d_se_v2_)
            cudaFree(d_se_v2_);
        size_t offsets_bytes = n_offsets_ * sizeof(int2);
        CUDA_CHECK(cudaMalloc(&d_se_v2_, offsets_bytes));
        CUDA_CHECK(cudaMemcpy(d_se_v2_, offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));
    }

    // 使用 OpenCV 进行参考计算
    cv::Mat computeOpenCVMorph(int op, const cv::Mat &kernel)
    {
        cv::Mat result;
        cv::morphologyEx(test_image_, result, op, kernel);
        return result;
    }

    // 使用 CUDA v0 进行计算
    cv::Mat computeCUDAv0Morph(int op)
    {
        morphologyEx<v0::DirectAccessStrategy, uint8_t>(d_input_, d_output_, d_tmp1_, d_tmp2_, img_w_, img_h_,
                                                        img_stride_, op, d_se_v1_, 0, se_w_, se_h_, anchor_x_,
                                                        anchor_y_, stream_);

        CUDA_CHECK(cudaStreamSynchronize(stream_));

        cv::Mat result(img_h_, img_w_, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(result.data, d_output_, img_h_ * img_stride_, cudaMemcpyDeviceToHost));
        return result;
    }

    // 使用 CUDA v1 进行计算
    cv::Mat computeCUDAv1Morph(int op)
    {
        morphologyEx<v1::SharedMemoryStrategy, uint8_t>(d_input_, d_output_, d_tmp1_, d_tmp2_, img_w_, img_h_,
                                                        img_stride_, op, d_se_v1_, 0, se_w_, se_h_, anchor_x_,
                                                        anchor_y_, stream_);

        CUDA_CHECK(cudaStreamSynchronize(stream_));

        cv::Mat result(img_h_, img_w_, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(result.data, d_output_, img_h_ * img_stride_, cudaMemcpyDeviceToHost));
        return result;
    }

    // 使用 CUDA v2 进行计算
    cv::Mat computeCUDAv2Morph(int op)
    {
        morphologyEx<v2::OffsetOptimizedStrategy, int2>(d_input_, d_output_, d_tmp1_, d_tmp2_, img_w_, img_h_,
                                                        img_stride_, op, d_se_v2_, n_offsets_, se_w_, se_h_, anchor_x_,
                                                        anchor_y_, stream_);

        CUDA_CHECK(cudaStreamSynchronize(stream_));

        cv::Mat result(img_h_, img_w_, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(result.data, d_output_, img_h_ * img_stride_, cudaMemcpyDeviceToHost));
        return result;
    }

    // 计算两个图像之间的差异
    double computeImageDifference(const cv::Mat &img1, const cv::Mat &img2)
    {
        cv::Mat diff;
        cv::absdiff(img1, img2, diff);
        cv::Scalar sum = cv::sum(diff);
        return sum[0] / (img1.rows * img1.cols);
    }

protected:
    cv::Mat test_image_;
    int     img_w_, img_h_, img_stride_;
    int     se_w_, se_h_, anchor_x_, anchor_y_;
    int     n_offsets_;

    uint8_t *d_input_  = nullptr;
    uint8_t *d_output_ = nullptr;
    uint8_t *d_tmp1_   = nullptr;
    uint8_t *d_tmp2_   = nullptr;
    uint8_t *d_se_v1_  = nullptr;
    int2    *d_se_v2_  = nullptr;

    cudaStream_t stream_ = nullptr;
    dim3         block_dim_;
};

// 测试用例
TEST_P(MorphologyCudaTest, CompareV1V2WithOpenCV)
{
    auto p   = GetParam();
    int  ksz = p.kernel_size / 2 * 2 + 1;

    std::vector<int> shapes = {cv::MORPH_RECT, cv::MORPH_CROSS, cv::MORPH_ELLIPSE};
    std::vector<int> operations
        = {cv::MORPH_DILATE, cv::MORPH_ERODE, cv::MORPH_OPEN, cv::MORPH_CLOSE, cv::MORPH_TOPHAT, cv::MORPH_BLACKHAT};

    for (int shape : shapes)
    {
        for (int op : operations)
        {
            cv::Mat kernel = cv::getStructuringElement(shape, cv::Size(ksz, ksz));
            prepareStructuringElement(kernel);
            cv::Mat opencv_result  = computeOpenCVMorph(op, kernel);
            cv::Mat cuda_v0_result = computeCUDAv1Morph(op);
            cv::Mat cuda_v1_result = computeCUDAv1Morph(op);
            cv::Mat cuda_v2_result = computeCUDAv2Morph(op);
            // 验证结果不为空且尺寸正确
            EXPECT_FALSE(opencv_result.empty());
            EXPECT_FALSE(cuda_v0_result.empty());
            EXPECT_FALSE(cuda_v1_result.empty());
            EXPECT_FALSE(cuda_v2_result.empty());
            EXPECT_EQ(opencv_result.size(), cuda_v0_result.size());
            EXPECT_EQ(opencv_result.size(), cuda_v1_result.size());
            EXPECT_EQ(opencv_result.size(), cuda_v2_result.size());
            // 比较结果
            double diff_v0 = computeImageDifference(opencv_result, cuda_v0_result);
            double diff_v1 = computeImageDifference(opencv_result, cuda_v1_result);
            double diff_v2 = computeImageDifference(opencv_result, cuda_v2_result);
            EXPECT_LT(diff_v0, 1.0) << "CUDA v0 difference too large, kernel size: " << p.kernel_size;
            EXPECT_LT(diff_v1, 1.0) << "CUDA v1 difference too large, kernel size: " << p.kernel_size;
            EXPECT_LT(diff_v2, 1.0) << "CUDA v2 difference too large, kernel size: " << p.kernel_size;
        }
    }
}

// clang-format off

// 参数化：测试多种操作

INSTANTIATE_TEST_SUITE_P(
    AllOps,
    MorphologyCudaTest,
    ::testing::Values(
        MorphTestParams{1},
        MorphTestParams{2},
        MorphTestParams{3},
        MorphTestParams{5},
        MorphTestParams{6},
        MorphTestParams{7},
        MorphTestParams{11},
        MorphTestParams{15},
        MorphTestParams{20},
        MorphTestParams{31}
    )
);

// clang-format on
