#include "morph_benchmark.h"

#include <benchmark/benchmark.h>

#include <iostream>

using namespace free_kick::cuda::benchmark;

// ==================== Benchmark 注册 ====================

BENCHMARK(BM_OpenCV_Dilate)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V0_Dilate)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V1_Dilate)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V2_Dilate)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(BM_OpenCV_Erode)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V0_Erode)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V1_Erode)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V2_Erode)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(BM_OpenCV_Open)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V0_Open)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V1_Open)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V2_Open)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(BM_OpenCV_Close)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V0_Close)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V1_Close)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V2_Close)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(BM_OpenCV_TopHat)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V0_TopHat)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V1_TopHat)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V2_TopHat)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(BM_OpenCV_BlackHat)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V0_BlackHat)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V1_BlackHat)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_CUDA_V2_BlackHat)->Unit(benchmark::kMicrosecond)->UseRealTime();

int main(int argc, char **argv)
{
    try
    {
        // 初始化测试数据
        std::cout << "Init benchmark test data..." << std::endl;
        InitializeBenchmarkData("F:/Projects/morph_test/2025_08_18/picture_1_2025_08_18_18_02_25_059.png", 5);
        std::cout << "Init Done!" << std::endl;

        // 运行 benchmark
        std::cout << "Run benchmark..." << std::endl;
        std::cout << "Config: \n"
                  << "\t Image size: " << g_benchmark_data->test_image.size
                  << " , Kernel(ELLIPSE): " << g_benchmark_data->kernel.size << std::endl;
        std::cout << "\t CUDA v1 vs CUDA v2 vs OpenCV" << std::endl;
        std::cout << "========================================" << std::endl;

        benchmark::Initialize(&argc, argv);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();

        // 清理测试数据
        std::cout << "Clean up test data..." << std::endl;
        free_kick::cuda::benchmark::CleanupBenchmarkData();

        std::cout << "Benchmark Done!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << std::endl;
        free_kick::cuda::benchmark::CleanupBenchmarkData();
        return 1;
    }
    catch (...)
    {
        std::cerr << "发生未知异常" << std::endl;
        free_kick::cuda::benchmark::CleanupBenchmarkData();
        return 1;
    }

    return 0;
}