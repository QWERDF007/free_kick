#include "morph_benchmark.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace free_kick::cuda::benchmark {

// 全局测试数据管理器
std::unique_ptr<MorphBenchmarkData> g_benchmark_data;

// ==================== MorphBenchmarkData 实现 ====================

MorphBenchmarkData::MorphBenchmarkData(const std::string &test_image_path, int kernel_size, int se_shape)
    : kernel_size(kernel_size)
    , se_shape(se_shape)
{
    // 尝试加载指定图片获取实际尺寸
    test_image = cv::imread(test_image_path, cv::IMREAD_GRAYSCALE);
    if (test_image.empty())
        return;

    // 使用图片的原始尺寸
    width  = test_image.cols;
    height = test_image.rows;
    stride = width;

    // 基于实际尺寸分配内存
    size_t img_bytes = height * stride * sizeof(uint8_t);

    // 分配 Device 内存
    cudaMalloc(&d_input, img_bytes);
    cudaMalloc(&d_output, img_bytes);
    cudaMalloc(&d_tmp1, img_bytes);
    cudaMalloc(&d_tmp2, img_bytes);

    cudaStreamCreate(&stream);

    cudaMemcpy(d_input, test_image.data, img_bytes, cudaMemcpyHostToDevice);

    // 生成结构元素
    getStructuringElement(kernel_size, se_shape);
}

MorphBenchmarkData::~MorphBenchmarkData()
{
    // 释放 Device 内存
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_tmp1);
    cudaFree(d_tmp2);
    if (d_se_v0)
        cudaFree(d_se_v0);
    if (d_se_v2)
        cudaFree(d_se_v2);
}

void MorphBenchmarkData::getStructuringElement(int _kernel_size, int _se_shape)
{
    // 更新成员变量
    this->kernel_size = _kernel_size;
    this->se_shape    = _se_shape;

    const int ksz = kernel_size / 2 * 2 + 1;
    kernel        = cv::getStructuringElement(se_shape, cv::Size(ksz, ksz));

    se_w     = kernel.cols;
    se_h     = kernel.rows;
    anchor_x = se_w / 2;
    anchor_y = se_h / 2;

    // 为 v0 准备（直接复制掩码）
    if (d_se_v0)
        cudaFree(d_se_v0);
    size_t se_bytes = se_w * se_h * sizeof(uint8_t);
    cudaMalloc(&d_se_v0, se_bytes);
    cudaMemcpy(d_se_v0, kernel.data, se_bytes, cudaMemcpyHostToDevice);

    // 为 v2 准备（构建偏移列表）
    auto offsets = free_kick::cuda::ops::buildOffsetList(kernel.data, se_w, se_h, anchor_x, anchor_y);
    n_offsets    = static_cast<int>(offsets.size());

    if (d_se_v2)
        cudaFree(d_se_v2);
    size_t offsets_bytes = n_offsets * sizeof(int2);
    cudaMalloc(&d_se_v2, offsets_bytes);
    cudaMemcpy(d_se_v2, offsets.data(), offsets_bytes, cudaMemcpyHostToDevice);
}

// ==================== 全局初始化函数 ====================

void InitializeBenchmarkData(const std::string &path, const int kernel_size, const int se_shape)
{
    g_benchmark_data = std::make_unique<MorphBenchmarkData>(path, kernel_size, se_shape);
}

void CleanupBenchmarkData()
{
    g_benchmark_data.reset();
}

} // namespace free_kick::cuda::benchmark

// ==================== OpenCV 版本模板函数实现 ====================

void BM_OpenCV_Morphology(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto &data = *free_kick::cuda::benchmark::g_benchmark_data;

    // 从 benchmark 参数中获取操作类型、kernel_size 和 se_shape
    int morph_op    = static_cast<int>(state.range(0));
    int kernel_size = static_cast<int>(state.range(1));
    int se_shape    = static_cast<int>(state.range(2));

    // 如果参数发生变化，重新生成结构元素
    if (kernel_size != data.kernel_size || se_shape != data.se_shape)
    {
        data.getStructuringElement(kernel_size, se_shape);
    }

    cv::Point anchor(data.anchor_x, data.anchor_y);

    for (auto _ : state)
    {
        cv::morphologyEx(data.test_image, data.output_image, morph_op, data.kernel, anchor);
    }
}