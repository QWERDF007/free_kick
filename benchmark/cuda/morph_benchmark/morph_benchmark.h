#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

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
    uint8_t *d_se_v1  = nullptr; // v1 使用 uint8_t 掩码
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

// CUDA v0 版本 benchmark 函数（使用新的统一接口）
void BM_CUDA_V0_Dilate(benchmark::State &state);
void BM_CUDA_V0_Erode(benchmark::State &state);
void BM_CUDA_V0_Open(benchmark::State &state);
void BM_CUDA_V0_Close(benchmark::State &state);
void BM_CUDA_V0_TopHat(benchmark::State &state);
void BM_CUDA_V0_BlackHat(benchmark::State &state);

// CUDA v1 版本 benchmark 函数（使用新的统一接口）
void BM_CUDA_V1_Dilate(benchmark::State &state);
void BM_CUDA_V1_Erode(benchmark::State &state);
void BM_CUDA_V1_Open(benchmark::State &state);
void BM_CUDA_V1_Close(benchmark::State &state);
void BM_CUDA_V1_TopHat(benchmark::State &state);
void BM_CUDA_V1_BlackHat(benchmark::State &state);

// CUDA v2 版本 benchmark 函数（使用新的统一接口）
void BM_CUDA_V2_Dilate(benchmark::State &state);
void BM_CUDA_V2_Erode(benchmark::State &state);
void BM_CUDA_V2_Open(benchmark::State &state);
void BM_CUDA_V2_Close(benchmark::State &state);
void BM_CUDA_V2_TopHat(benchmark::State &state);
void BM_CUDA_V2_BlackHat(benchmark::State &state);

// OpenCV 版本 benchmark 函数
void BM_OpenCV_Dilate(benchmark::State &state);
void BM_OpenCV_Erode(benchmark::State &state);
void BM_OpenCV_Open(benchmark::State &state);
void BM_OpenCV_Close(benchmark::State &state);
void BM_OpenCV_TopHat(benchmark::State &state);
void BM_OpenCV_BlackHat(benchmark::State &state);