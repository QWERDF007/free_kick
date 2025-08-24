#include "morph_benchmark.h"

#include <benchmark/benchmark.h>

#include <iostream>

using namespace free_kick::cuda::benchmark;

// ==================== Benchmark 注册 ====================

// 定义形态学操作类型
const std::vector<int> morph_ops = {
    cv::MORPH_DILATE,  // 膨胀
    cv::MORPH_ERODE,   // 腐蚀
    cv::MORPH_OPEN,    // 开运算
    cv::MORPH_CLOSE,   // 闭运算
    cv::MORPH_TOPHAT,  // 顶帽
    cv::MORPH_BLACKHAT // 黑帽
};

// 操作名称映射
const std::vector<std::string> morph_op_names = {"Dilate", "Erode", "Open", "Close", "TopHat", "BlackHat"};

// 注册所有 benchmark
void RegisterBenchmarks()
{
    for (size_t i = 0; i < morph_ops.size(); ++i)
    {
        int         morph_op = morph_ops[i];
        std::string op_name  = morph_op_names[i];

        // OpenCV 版本
        benchmark::RegisterBenchmark(("OpenCV_" + op_name).c_str(),
                                     [morph_op](benchmark::State &state)
                                     {
                                         //  state.SetLabel("OpenCV " + morph_op_names[morph_op]);
                                         BM_OpenCV_Morphology(state);
                                     })
            ->Unit(benchmark::kMicrosecond)
            ->UseRealTime()
            ->Arg(morph_op);

        // CUDA V0 版本
        benchmark::RegisterBenchmark(
            ("CUDA_V0_" + op_name).c_str(),
            [morph_op](benchmark::State &state)
            {
                // state.SetLabel("CUDA V0 " + morph_op_names[morph_op]);
                BM_CUDA_Morphology<free_kick::cuda::ops::v0::DirectAccessExecutor<uint8_t>, uint8_t>(state);
            })
            ->Unit(benchmark::kMicrosecond)
            ->UseRealTime()
            ->Arg(morph_op);

        // CUDA V1 版本
        benchmark::RegisterBenchmark(
            ("CUDA_V1_" + op_name).c_str(),
            [morph_op](benchmark::State &state)
            {
                // state.SetLabel("CUDA V1 " + morph_op_names[morph_op]);
                BM_CUDA_Morphology<free_kick::cuda::ops::v1::SharedMemoryExecutor<uint8_t>, uint8_t>(state);
            })
            ->Unit(benchmark::kMicrosecond)
            ->UseRealTime()
            ->Arg(morph_op);

        // CUDA V2 版本
        benchmark::RegisterBenchmark(
            ("CUDA_V2_" + op_name).c_str(),
            [morph_op](benchmark::State &state)
            {
                // state.SetLabel("CUDA V2 " + morph_op_names[morph_op]);
                BM_CUDA_Morphology<free_kick::cuda::ops::v2::OffsetOptimizedExecutor<uint8_t>, int2>(state);
            })
            ->Unit(benchmark::kMicrosecond)
            ->UseRealTime()
            ->Arg(morph_op);
    }
}

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
        std::cout << "\t CUDA v0 vs CUDA v1 vs CUDA v2 vs OpenCV" << std::endl;
        std::cout << "========================================" << std::endl;

        benchmark::Initialize(&argc, argv);

        // 注册所有 benchmark
        RegisterBenchmarks();

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