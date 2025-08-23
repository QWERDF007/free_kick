#include "common/utility.h"

#include "morphology_unified.cuh"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace free_kick::cuda::ops::unified;

class MorphologyCudaPerformTest
{
public:
    void SetUp(const cv::Mat &src)
    {
        test_image_ = src.clone();
        test_out_   = cv::Mat(test_image_.size(), CV_8UC1);
        img_w_      = test_image_.cols;
        img_h_      = test_image_.rows;
        img_stride_ = static_cast<int>(test_image_.step[0]);

        // 分配 CUDA 内存
        img_bytes_ = img_h_ * img_stride_;
        CUDA_CHECK(cudaMalloc(&d_input_, img_bytes_));
        CUDA_CHECK(cudaMalloc(&d_output_, img_bytes_));
        CUDA_CHECK(cudaMalloc(&d_tmp1_, img_bytes_));
        CUDA_CHECK(cudaMalloc(&d_tmp2_, img_bytes_));

        // 创建 CUDA stream
        CUDA_CHECK(cudaStreamCreate(&stream_));
        // 创建 CUDA event
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        CUDA_CHECK(cudaEventCreate(&ev_kernel_start));
        CUDA_CHECK(cudaEventCreate(&ev_kernel_stop));
        CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
        CUDA_CHECK(cudaEventCreate(&ev_h2d_stop));
        CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
        CUDA_CHECK(cudaEventCreate(&ev_d2h_stop));
        // 设置默认 block 维度
        block_dim_ = dim3(32, 16);
    }

    void TearDown()
    {
        CUDA_CHECK(cudaFree(d_input_));
        CUDA_CHECK(cudaFree(d_output_));
        CUDA_CHECK(cudaFree(d_tmp1_));
        CUDA_CHECK(cudaFree(d_tmp2_));
        CUDA_CHECK(cudaFree(d_se_mask_));
        CUDA_CHECK(cudaStreamDestroy(stream_));

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        CUDA_CHECK(cudaEventDestroy(ev_kernel_start));
        CUDA_CHECK(cudaEventDestroy(ev_kernel_stop));
        CUDA_CHECK(cudaEventDestroy(ev_h2d_start));
        CUDA_CHECK(cudaEventDestroy(ev_h2d_stop));
        CUDA_CHECK(cudaEventDestroy(ev_d2h_start));
        CUDA_CHECK(cudaEventDestroy(ev_d2h_stop));
    }

    // 准备结构元素（为统一接口）
    void prepareStructuringElement(const int shape, const int kernel_size)
    {
        const int ksz = kernel_size / 2 * 2 + 1;
        kernel_       = cv::getStructuringElement(shape, cv::Size(ksz, ksz));

        se_w_     = kernel_.cols;
        se_h_     = kernel_.rows;
        anchor_x_ = se_w_ / 2;
        anchor_y_ = se_h_ / 2;

        // 为统一接口准备设备端结构元素掩码
        if (d_se_mask_)
            cudaFree(d_se_mask_);
        size_t se_bytes = se_w_ * se_h_ * sizeof(uint8_t);
        CUDA_CHECK(cudaMalloc(&d_se_mask_, se_bytes));
        CUDA_CHECK(cudaMemcpy(d_se_mask_, kernel_.data, se_bytes, cudaMemcpyHostToDevice));
    }

    template<typename Func>
    std::vector<double> measureOpenCVExecutionTime(Func &&func)
    {
        float  kernel_ms = 0.0f;
        float  cv_ms     = 0.0f;
        float  h2d_ms    = 0.0f; // OpenCV不需要h2d传输
        float  d2h_ms    = 0.0f; // OpenCV不需要d2h传输
        double wall_ms   = 0.0;

        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end  = std::chrono::high_resolution_clock::now();
        wall_ms   = std::chrono::duration<double, std::milli>(end - start).count();
        cv_ms     = static_cast<float>(wall_ms); // OpenCV的总CUDA时间等于wall时间
        kernel_ms = static_cast<float>(wall_ms); // OpenCV的kernel时间等于wall时间

        std::vector<double> times = {h2d_ms, d2h_ms, kernel_ms, cv_ms, wall_ms};
        return times;
    }

    // 计算执行时间（毫秒）, [h2d, d2h, kernel, cuda_total, wall]
    template<typename Func>
    std::vector<double> measureCudaExecutionTime(Func &&func)
    {
        float  kernel_ms = 0.0f;
        float  cuda_ms   = 0.0f;
        float  h2d_ms    = 0.0f;
        float  d2h_ms    = 0.0f;
        double wall_ms   = 0.0;

        auto start = std::chrono::high_resolution_clock::now();
        // 注册主机内存以启用异步传输（在计时之外）
        CUDA_CHECK(cudaHostRegister(const_cast<uint8_t *>(test_image_.data), img_bytes_, cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(const_cast<uint8_t *>(test_out_.data), img_bytes_, cudaHostRegisterDefault));
        // 事件开始
        CUDA_CHECK(cudaEventRecord(ev_start, stream_));
        // 异步传输输入数据到设备, 记录事件
        CUDA_CHECK(cudaEventRecord(ev_h2d_start, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_input_, test_image_.data, img_bytes_, cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaEventRecord(ev_h2d_stop, stream_));
        CUDA_CHECK(cudaEventRecord(ev_kernel_start, stream_));
        func();
        CUDA_CHECK(cudaEventRecord(ev_kernel_stop, stream_));
        // 异步传输输入数据到主机
        CUDA_CHECK(cudaEventRecord(ev_d2h_start, stream_));
        CUDA_CHECK(cudaMemcpyAsync(test_out_.data, d_output_, img_bytes_, cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaEventRecord(ev_d2h_stop, stream_));
        // 事件结束, 等待同步
        CUDA_CHECK(cudaEventRecord(ev_stop, stream_));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        // 取消注册主机内存（在计时之外）
        CUDA_CHECK(cudaHostUnregister(test_image_.data));
        CUDA_CHECK(cudaHostUnregister(test_out_.data));

        auto end = std::chrono::high_resolution_clock::now();
        wall_ms  = std::chrono::duration<double, std::milli>(end - start).count();

        CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, ev_start, ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_kernel_start, ev_kernel_stop));
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, ev_h2d_start, ev_h2d_stop));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, ev_d2h_start, ev_d2h_stop));

        std::vector<double> times = {h2d_ms, d2h_ms, kernel_ms, cuda_ms, wall_ms};
        return times;
    }

    void run(const int shape, const int op, const int kernel_size)
    {
        // 获取结构元素 (kernel)
        prepareStructuringElement(shape, kernel_size);

        auto cv_times
            = measureOpenCVExecutionTime([this, op] { cv::morphologyEx(test_image_, test_out_, op, kernel_); });

        auto cuda_v0_times = measureCudaExecutionTime(
            [this, op]
            {
                morphologyEx<DirectAccessStrategy, uint8_t>(d_input_, d_output_, d_tmp1_, d_tmp2_, img_w_, img_h_,
                                                            img_stride_, op, d_se_mask_, 0, se_w_, se_h_, anchor_x_,
                                                            anchor_y_, stream_);
            });

        auto cuda_v1_times = measureCudaExecutionTime(
            [this, op]
            {
                morphologyEx<SharedMemoryStrategy, uint8_t>(d_input_, d_output_, d_tmp1_, d_tmp2_, img_w_, img_h_,
                                                            img_stride_, op, d_se_mask_, 0, se_w_, se_h_, anchor_x_,
                                                            anchor_y_, stream_);
            });

        auto cuda_v2_times = measureCudaExecutionTime(
            [this, op]
            {
                // 为v2策略准备偏移列表
                auto offsets
                    = free_kick::cuda::ops::unified::buildOffsetList(kernel_.data, se_w_, se_h_, anchor_x_, anchor_y_);
                int2 *d_offsets;
                CUDA_CHECK(cudaMalloc(&d_offsets, offsets.size() * sizeof(int2)));
                CUDA_CHECK(cudaMemcpyAsync(d_offsets, offsets.data(), offsets.size() * sizeof(int2),
                                           cudaMemcpyHostToDevice, stream_));

                morphologyEx<OffsetOptimizedStrategy, int2>(
                    d_input_, d_output_, d_tmp1_, d_tmp2_, img_w_, img_h_, img_stride_, op, d_offsets,
                    static_cast<int>(offsets.size()), se_w_, se_h_, anchor_x_, anchor_y_, stream_);

                CUDA_CHECK(cudaFree(d_offsets));
            });

        // 获取操作名称
        std::string op_name;
        switch (op)
        {
        case cv::MORPH_DILATE:
            op_name = "DILATE";
            break;
        case cv::MORPH_ERODE:
            op_name = "ERODE";
            break;
        case cv::MORPH_OPEN:
            op_name = "OPEN";
            break;
        case cv::MORPH_CLOSE:
            op_name = "CLOSE";
            break;
        case ::cv::MORPH_BLACKHAT:
            op_name = "BLACKHAT";
            break;
        case ::cv::MORPH_TOPHAT:
            op_name = "TOPKHAT";
            break;
        default:
            op_name = "UNKNOWN";
            break;
        }

        std::vector<std::pair<std::string, std::vector<double>>> names_times{
            {"cv",      cv_times},
            {"v0", cuda_v0_times},
            {"v1", cuda_v1_times},
            {"v2", cuda_v2_times},
        };

        // 提取时间数据 [h2d_ms, d2h_ms, kernel_ms, cuda_total_ms, wall_ms]
        const size_t last         = cv_times.size() - 1;
        double       cv_wall_time = cv_times[last];

        // 打印格式化输出

        std::cout << std::fixed << std::setprecision(3);
        for (const auto &[name, times] : names_times)
        {
            double scale = cv_wall_time / (times[last] + 1e-9);
            std::cout << "| " << op_name << " | " << name << " | " << times[0] << " | " << times[1] << " | " << times[2]
                      << " | " << times[3] << " | " << times[4] << " | " << scale << "x |" << std::endl;
        }
    }

protected:
    cv::Mat  test_image_;
    cv::Mat  test_out_;
    cv::Mat  kernel_;
    size_t   img_bytes_;
    int      img_w_, img_h_, img_stride_;
    int      se_w_, se_h_, anchor_x_, anchor_y_;
    uint8_t *d_input_   = nullptr;
    uint8_t *d_output_  = nullptr;
    uint8_t *d_tmp1_    = nullptr;
    uint8_t *d_tmp2_    = nullptr;
    uint8_t *d_se_mask_ = nullptr;

    // CUDA 事件计时器
    cudaEvent_t ev_start, ev_stop;
    cudaEvent_t ev_kernel_start, ev_kernel_stop;
    cudaEvent_t ev_h2d_start, ev_h2d_stop;
    cudaEvent_t ev_d2h_start, ev_d2h_stop;

    cudaStream_t stream_ = nullptr;
    dim3         block_dim_;
};

int main(int argc, char **argv)
{
    const char *input_path  = (argc > 1) ? argv[1] : "input.png";
    int         kernel_size = (argc > 2) ? std::max(1, atoi(argv[2])) : 3; // 结构元素大小

    // 读入灰度图
    cv::Mat img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return 1;
    }
    MorphologyCudaPerformTest runner;
    runner.SetUp(img);
    // std::vector<int> shapes     = {cv::MORPH_RECT, cv::MORPH_CROSS, cv::MORPH_ELLIPSE};
    std::vector<int> operations
        = {cv::MORPH_DILATE, cv::MORPH_ERODE, cv::MORPH_OPEN, cv::MORPH_CLOSE, cv::MORPH_TOPHAT, cv::MORPH_BLACKHAT};

    std::cout << "Image size: " << img.size << std::endl;
    std::cout << "| OP | version | h2d time | d2h time | kernel time | cuda time | wall time | Speed up |" << std::endl;
    std::cout << "| ---- | ---- | ---- |---- | ---- | ---- | ---- | ---- |" << std::endl;

    for (int op : operations)
    {
        runner.run(cv::MORPH_ELLIPSE, op, kernel_size);
    }
    runner.TearDown();
    return 0;
}
