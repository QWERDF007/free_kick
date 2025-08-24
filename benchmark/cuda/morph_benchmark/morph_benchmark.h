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
    MorphBenchmarkData(const std::string &img_path, int se_size);
    ~MorphBenchmarkData();

    // 禁用拷贝构造和赋值
    MorphBenchmarkData(const MorphBenchmarkData &)            = delete;
    MorphBenchmarkData &operator=(const MorphBenchmarkData &) = delete;

    void getStructuringElement();

    int kernel_size;
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
    uint8_t *d_se_v0  = nullptr; // v0 使用 uint8_t 掩码
    int2    *d_se_v2  = nullptr; // v2 使用 int2 偏移

    dim3         block_dim;
    cudaStream_t stream = nullptr;
};

// 全局测试数据管理器
extern std::unique_ptr<MorphBenchmarkData> g_benchmark_data;

// 初始化函数
void InitializeBenchmarkData(const std::string &img_path, int kernel_size);
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

    // 从 benchmark 参数中获取操作类型
    int morph_op = state.range(0);

    // 根据执行器类型选择合适的结构元素数据
    SEType *d_se;
    int     n_offsets;

    if constexpr (std::is_same_v<SEType, uint8_t>) // v0、v1
    {
        d_se      = data.d_se_v0;
        n_offsets = data.se_w * data.se_h;
    }
    else
    {
        // v2 使用 int2 偏移
        d_se      = data.d_se_v2;
        n_offsets = data.n_offsets;
    }

    for (auto _ : state)
    {
        free_kick::cuda::ops::morphologyEx<Executor, SEType>(
            data.d_input, data.d_output, data.d_tmp1, data.d_tmp2, data.width, data.height, data.stride, morph_op, d_se,
            n_offsets, data.se_w, data.se_h, data.anchor_x, data.anchor_y, data.stream);
        cudaStreamSynchronize(data.stream);
    }

    // // 为膨胀操作添加性能计数器
    // // if (morph_op == cv::MORPH_DILATE)
    // {
    //     double pixels_processed      = static_cast<double>(data.width * data.height);
    //     state.counters["pixels/sec"] = benchmark::Counter(pixels_processed, benchmark::Counter::kIsRate);
    //     state.counters["MPix/sec"]   = benchmark::Counter(pixels_processed / 1e6, benchmark::Counter::kIsRate);
    // }
}

// ==================== OpenCV 版本模板函数 ====================

// OpenCV 形态学操作模板函数
void BM_OpenCV_Morphology(benchmark::State &state);