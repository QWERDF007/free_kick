#pragma once

#include "morphology_unified.cuh"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace free_kick::cuda::benchmark {

// 测试数据管理类
class MorphBenchmarkData
{
public:
    MorphBenchmarkData(const std::string &img_path);
    ~MorphBenchmarkData();

    // 禁用拷贝构造和赋值
    MorphBenchmarkData(const MorphBenchmarkData &)            = delete;
    MorphBenchmarkData &operator=(const MorphBenchmarkData &) = delete;

    void getStructuringElement(int kernel_size, int se_shape);

    int kernel_size{0};
    int se_shape{0}; // 结构元素形状
    int width, height, stride;
    int se_w, se_h;
    int anchor_x, anchor_y;
    int n_offsets;

    cv::Mat test_image;
    cv::Mat output_image;
    cv::Mat kernel;

    // Device 内存
    uint8_t *d_input  = nullptr;
    uint8_t *d_output = nullptr;
    uint8_t *d_tmp1   = nullptr;
    uint8_t *d_tmp2   = nullptr;
    uint8_t *d_se_u8  = nullptr; // v0 使用 uint8_t 掩码
    int2    *d_se_i2  = nullptr; // v2 使用 int2 偏移

    dim3         block_dim;
    cudaStream_t stream = nullptr;
};

// 全局测试数据管理器
extern std::unique_ptr<MorphBenchmarkData> g_benchmark_data;

// 初始化函数
void InitializeBenchmarkData(const std::string &img_path);
void CleanupBenchmarkData();

} // namespace free_kick::cuda::benchmark

// ==================== CUDA 版本模板函数 ====================

// CUDA 形态学操作模板函数
template<typename Executor, typename SEType>
void BM_CUDA_Morphology(benchmark::State &state)
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

    // 根据执行器类型选择合适的结构元素数据
    SEType *d_se;
    int     n_offsets;

    if constexpr (std::is_same_v<SEType, uint8_t>) // v0、v1、v2、v3
    {
        d_se      = data.d_se_u8;
        n_offsets = 0; // v0-v3不使用偏移列表，n_offsets设为0
    }
    else
    {
        // v4 使用 int2 偏移列表
        d_se      = data.d_se_i2;
        n_offsets = data.n_offsets;
    }

    for (auto _ : state)
    {
        free_kick::cuda::ops::morphologyEx<Executor, SEType>(
            data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width, data.height, data.stride, morph_op, d_se,
            n_offsets, data.se_w, data.se_h, data.anchor_x, data.anchor_y, data.stream);
        cudaStreamSynchronize(data.stream);
    }
}

// ==================== OpenCV 版本模板函数 ====================

// OpenCV 形态学操作模板函数
void BM_OpenCV_Morphology(benchmark::State &state);