#include "morph_benchmark.h"

#include "morphology_cuda_v1.h"
#include "morphology_cuda_v2.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace free_kick::cuda::benchmark {

// 全局测试数据管理器
std::unique_ptr<MorphBenchmarkData> g_benchmark_data;

// ==================== MorphBenchmarkData 实现 ====================

MorphBenchmarkData::MorphBenchmarkData(const std::string &test_image_path, int kernel_size)
    : kernel_size(kernel_size)
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
    block_dim = dim3(32, 16);

    cudaMemcpy(d_input, test_image.data, img_bytes, cudaMemcpyHostToDevice);

    // 生成结构元素
    getStructuringElement();
}

MorphBenchmarkData::~MorphBenchmarkData()
{
    // 释放 Device 内存
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_tmp1);
    cudaFree(d_tmp2);
    if (d_se_v1)
        cudaFree(d_se_v1);
    if (d_se_v2)
        cudaFree(d_se_v2);
}

void MorphBenchmarkData::getStructuringElement()
{
    const int ksz = kernel_size / 2 * 2 + 1;
    kernel        = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksz, ksz));

    se_w     = kernel.cols;
    se_h     = kernel.rows;
    anchor_x = se_w / 2;
    anchor_y = se_h / 2;

    // 为 v1 准备（直接复制掩码）
    if (d_se_v1)
        cudaFree(d_se_v1);
    size_t se_bytes = se_w * se_h * sizeof(uint8_t);
    cudaMalloc(&d_se_v1, se_bytes);
    cudaMemcpy(d_se_v1, kernel.data, se_bytes, cudaMemcpyHostToDevice);

    // 为 v2 准备（构建偏移列表）
    auto offsets = free_kick::cuda::ops::v2::build_se_offsets(kernel.data, se_w, se_h, anchor_x, anchor_y);
    n_offsets    = static_cast<int>(offsets.size());

    if (d_se_v2)
        cudaFree(d_se_v2);
    size_t offsets_bytes = n_offsets * sizeof(int2);
    cudaMalloc(&d_se_v2, offsets_bytes);
    cudaMemcpy(d_se_v2, offsets.data(), offsets_bytes, cudaMemcpyHostToDevice);
}

// ==================== 全局初始化函数 ====================

void InitializeBenchmarkData(const std::string &path, const int kernel_size)
{
    g_benchmark_data = std::make_unique<MorphBenchmarkData>(path, kernel_size);
}

void CleanupBenchmarkData()
{
    g_benchmark_data.reset();
}

} // namespace free_kick::cuda::benchmark

// ==================== CUDA V1 Benchmark 函数 ====================

void BM_CUDA_V1_Dilate(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto &data = *free_kick::cuda::benchmark::g_benchmark_data;

    for (auto _ : state)
    {
        free_kick::cuda::ops::v1::morphologyEx(data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width,
                                               data.height, data.stride, cv::MORPH_DILATE, data.d_se_v1, data.se_w,
                                               data.se_h, data.anchor_x, data.anchor_y, data.block_dim, data.stream);
        cudaStreamSynchronize(data.stream);
    }

    double pixels_processed      = static_cast<double>(data.width * data.height);
    state.counters["pixels/sec"] = benchmark::Counter(pixels_processed, benchmark::Counter::kIsRate);
    state.counters["MPix/sec"]   = benchmark::Counter(pixels_processed / 1e6, benchmark::Counter::kIsRate);
}

void BM_CUDA_V1_Erode(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto &data = *free_kick::cuda::benchmark::g_benchmark_data;

    for (auto _ : state)
    {
        free_kick::cuda::ops::v1::morphologyEx(data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width,
                                               data.height, data.stride, cv::MORPH_ERODE, data.d_se_v1, data.se_w,
                                               data.se_h, data.anchor_x, data.anchor_y, data.block_dim, data.stream);
        cudaStreamSynchronize(data.stream);
    }
}

void BM_CUDA_V1_Open(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto &data = *free_kick::cuda::benchmark::g_benchmark_data;

    for (auto _ : state)
    {
        free_kick::cuda::ops::v1::morphologyEx(data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width,
                                               data.height, data.stride, cv::MORPH_OPEN, data.d_se_v1, data.se_w,
                                               data.se_h, data.anchor_x, data.anchor_y, data.block_dim, data.stream);
        cudaStreamSynchronize(data.stream);
    }
}

void BM_CUDA_V1_Close(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto &data = *free_kick::cuda::benchmark::g_benchmark_data;

    for (auto _ : state)
    {
        free_kick::cuda::ops::v1::morphologyEx(data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width,
                                               data.height, data.stride, cv::MORPH_CLOSE, data.d_se_v1, data.se_w,
                                               data.se_h, data.anchor_x, data.anchor_y, data.block_dim, data.stream);
        cudaStreamSynchronize(data.stream);
    }
}

void BM_CUDA_V1_TopHat(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto &data = *free_kick::cuda::benchmark::g_benchmark_data;

    for (auto _ : state)
    {
        free_kick::cuda::ops::v1::morphologyEx(data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width,
                                               data.height, data.stride, cv::MORPH_TOPHAT, data.d_se_v1, data.se_w,
                                               data.se_h, data.anchor_x, data.anchor_y, data.block_dim, data.stream);
        cudaStreamSynchronize(data.stream);
    }
}

void BM_CUDA_V1_BlackHat(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto &data = *free_kick::cuda::benchmark::g_benchmark_data;

    for (auto _ : state)
    {
        free_kick::cuda::ops::v1::morphologyEx(data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width,
                                               data.height, data.stride, cv::MORPH_BLACKHAT, data.d_se_v1, data.se_w,
                                               data.se_h, data.anchor_x, data.anchor_y, data.block_dim, data.stream);
        cudaStreamSynchronize(data.stream);
    }
}

// ==================== CUDA V2 Benchmark 函数 ====================

void BM_CUDA_V2_Dilate(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto &data = *free_kick::cuda::benchmark::g_benchmark_data;

    for (auto _ : state)
    {
        free_kick::cuda::ops::v2::morphologyEx(data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width,
                                               data.height, data.stride, cv::MORPH_DILATE, data.d_se_v2, data.n_offsets,
                                               data.se_w, data.se_h, data.anchor_x, data.anchor_y, data.block_dim,
                                               data.stream);
        cudaStreamSynchronize(data.stream);
    }
}

void BM_CUDA_V2_Erode(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto &data = *free_kick::cuda::benchmark::g_benchmark_data;

    for (auto _ : state)
    {
        free_kick::cuda::ops::v2::morphologyEx(data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width,
                                               data.height, data.stride, cv::MORPH_ERODE, data.d_se_v2, data.n_offsets,
                                               data.se_w, data.se_h, data.anchor_x, data.anchor_y, data.block_dim,
                                               data.stream);
        cudaStreamSynchronize(data.stream);
    }
}

void BM_CUDA_V2_Open(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto &data = *free_kick::cuda::benchmark::g_benchmark_data;

    for (auto _ : state)
    {
        free_kick::cuda::ops::v2::morphologyEx(data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width,
                                               data.height, data.stride, cv::MORPH_OPEN, data.d_se_v2, data.n_offsets,
                                               data.se_w, data.se_h, data.anchor_x, data.anchor_y, data.block_dim,
                                               data.stream);
        cudaStreamSynchronize(data.stream);
    }
}

void BM_CUDA_V2_Close(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto &data = *free_kick::cuda::benchmark::g_benchmark_data;

    for (auto _ : state)
    {
        free_kick::cuda::ops::v2::morphologyEx(data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width,
                                               data.height, data.stride, cv::MORPH_CLOSE, data.d_se_v2, data.n_offsets,
                                               data.se_w, data.se_h, data.anchor_x, data.anchor_y, data.block_dim,
                                               data.stream);
        cudaStreamSynchronize(data.stream);
    }
}

void BM_CUDA_V2_TopHat(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto &data = *free_kick::cuda::benchmark::g_benchmark_data;

    for (auto _ : state)
    {
        free_kick::cuda::ops::v2::morphologyEx(data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width,
                                               data.height, data.stride, cv::MORPH_TOPHAT, data.d_se_v2, data.n_offsets,
                                               data.se_w, data.se_h, data.anchor_x, data.anchor_y, data.block_dim,
                                               data.stream);
        cudaStreamSynchronize(data.stream);
    }
}

void BM_CUDA_V2_BlackHat(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto &data = *free_kick::cuda::benchmark::g_benchmark_data;

    for (auto _ : state)
    {
        free_kick::cuda::ops::v2::morphologyEx(data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width,
                                               data.height, data.stride, cv::MORPH_BLACKHAT, data.d_se_v2,
                                               data.n_offsets, data.se_w, data.se_h, data.anchor_x, data.anchor_y,
                                               data.block_dim, data.stream);
        cudaStreamSynchronize(data.stream);
    }
}

// ==================== OpenCV Benchmark 函数 ====================

void BM_OpenCV_Dilate(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto     &data = *free_kick::cuda::benchmark::g_benchmark_data;
    cv::Point anchor(data.anchor_x, data.anchor_y);

    for (auto _ : state)
    {
        cv::morphologyEx(data.test_image, data.output_image, cv::MORPH_DILATE, data.kernel, anchor);
    }
}

void BM_OpenCV_Erode(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto     &data = *free_kick::cuda::benchmark::g_benchmark_data;
    cv::Point anchor(data.anchor_x, data.anchor_y);

    for (auto _ : state)
    {
        cv::erode(data.test_image, data.output_image, data.kernel, anchor);
        cv::morphologyEx(data.test_image, data.output_image, cv::MORPH_ERODE, data.kernel, anchor);
    }
}

void BM_OpenCV_Open(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto     &data = *free_kick::cuda::benchmark::g_benchmark_data;
    cv::Point anchor(data.anchor_x, data.anchor_y);

    for (auto _ : state)
    {
        cv::morphologyEx(data.test_image, data.output_image, cv::MORPH_OPEN, data.kernel, anchor);
    }
}

void BM_OpenCV_Close(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto     &data = *free_kick::cuda::benchmark::g_benchmark_data;
    cv::Point anchor(data.anchor_x, data.anchor_y);

    for (auto _ : state)
    {
        cv::morphologyEx(data.test_image, data.output_image, cv::MORPH_CLOSE, data.kernel, anchor);
    }
}

void BM_OpenCV_TopHat(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto     &data = *free_kick::cuda::benchmark::g_benchmark_data;
    cv::Point anchor(data.anchor_x, data.anchor_y);

    for (auto _ : state)
    {
        cv::morphologyEx(data.test_image, data.output_image, cv::MORPH_TOPHAT, data.kernel, anchor);
    }
}

void BM_OpenCV_BlackHat(benchmark::State &state)
{
    if (!free_kick::cuda::benchmark::g_benchmark_data)
    {
        state.SkipWithError("Benchmark data not initialized");
        return;
    }

    auto     &data = *free_kick::cuda::benchmark::g_benchmark_data;
    cv::Point anchor(data.anchor_x, data.anchor_y);

    for (auto _ : state)
    {
        cv::morphologyEx(data.test_image, data.output_image, cv::MORPH_BLACKHAT, data.kernel, anchor);
    }
}