#include "morph_benchmark.h"

#include <benchmark/benchmark.h>

#include <iostream>

using namespace free_kick::cuda::benchmark;

// ==================== Benchmark 注册 ====================

// 定义形态学操作类型
const std::vector<int> morph_ops = {
    cv::MORPH_ERODE,   // 腐蚀
    cv::MORPH_DILATE,  // 膨胀
    cv::MORPH_OPEN,    // 开运算
    cv::MORPH_CLOSE,   // 闭运算
    cv::MORPH_TOPHAT,  // 顶帽
    cv::MORPH_BLACKHAT // 黑帽
};

// 操作名称映射
const std::vector<std::string> morph_op_names = {"Erode", "Dilate", "Open", "Close", "TopHat", "BlackHat"};

// 定义结构元素形状
const std::vector<int> se_shapes = {
    cv::MORPH_RECT,    // 矩形
    cv::MORPH_ELLIPSE, // 椭圆形
    cv::MORPH_CROSS    // 十字形
};

// 形状名称映射
const std::vector<std::string> se_shape_names = {"RECT", "ELLIPSE", "CROSS"};

// 定义 kernel 大小
const std::vector<int> kernel_sizes = {3, 5, 7, 9, 11, 15, 21, 31};

// 注册所有 benchmark
void RegisterBenchmarks()
{
    for (size_t i = 0; i < morph_ops.size(); ++i)
    {
        int         morph_op = morph_ops[i];
        std::string op_name  = morph_op_names[i];

        for (size_t j = 0; j < se_shapes.size(); ++j)
        {
            int         se_shape   = se_shapes[j];
            std::string shape_name = se_shape_names[j];

            for (size_t k = 0; k < kernel_sizes.size(); ++k)
            {
                int kernel_size = kernel_sizes[k];

                // OpenCV 版本
                benchmark::RegisterBenchmark(
                    ("OpenCV_" + op_name + "_" + shape_name + "_" + std::to_string(kernel_size)).c_str(),
                    [morph_op, kernel_size, se_shape](benchmark::State &state) { BM_OpenCV_Morphology(state); })
                    ->Unit(benchmark::kMicrosecond)
                    ->UseRealTime()
                    ->Args({morph_op, kernel_size, se_shape});

                // CUDA V0 版本
                benchmark::RegisterBenchmark(
                    ("CUDA_V0_" + op_name + "_" + shape_name + "_" + std::to_string(kernel_size)).c_str(),
                    [morph_op, kernel_size, se_shape](benchmark::State &state)
                    { BM_CUDA_Morphology<free_kick::cuda::ops::v0<uint8_t>, uint8_t>(state); })
                    ->Unit(benchmark::kMicrosecond)
                    ->UseRealTime()
                    ->Args({morph_op, kernel_size, se_shape});

                // CUDA V1 版本
                benchmark::RegisterBenchmark(
                    ("CUDA_V1_" + op_name + "_" + shape_name + "_" + std::to_string(kernel_size)).c_str(),
                    [morph_op, kernel_size, se_shape](benchmark::State &state)
                    { BM_CUDA_Morphology<free_kick::cuda::ops::v1<uint8_t>, uint8_t>(state); })
                    ->Unit(benchmark::kMicrosecond)
                    ->UseRealTime()
                    ->Args({morph_op, kernel_size, se_shape});

                // CUDA V2 版本
                benchmark::RegisterBenchmark(
                    ("CUDA_V2_" + op_name + "_" + shape_name + "_" + std::to_string(kernel_size)).c_str(),
                    [morph_op, kernel_size, se_shape](benchmark::State &state)
                    { BM_CUDA_Morphology<free_kick::cuda::ops::v2<uint8_t>, uint8_t>(state); })
                    ->Unit(benchmark::kMicrosecond)
                    ->UseRealTime()
                    ->Args({morph_op, kernel_size, se_shape});

                // CUDA V3 版本
                benchmark::RegisterBenchmark(
                    ("CUDA_V3_" + op_name + "_" + shape_name + "_" + std::to_string(kernel_size)).c_str(),
                    [morph_op, kernel_size, se_shape](benchmark::State &state)
                    { BM_CUDA_Morphology<free_kick::cuda::ops::v3<uint8_t>, uint8_t>(state); })
                    ->Unit(benchmark::kMicrosecond)
                    ->UseRealTime()
                    ->Args({morph_op, kernel_size, se_shape});

                // CUDA V4 版本
                benchmark::RegisterBenchmark(
                    ("CUDA_V4_" + op_name + "_" + shape_name + "_" + std::to_string(kernel_size)).c_str(),
                    [morph_op, kernel_size, se_shape](benchmark::State &state)
                    { BM_CUDA_Morphology<free_kick::cuda::ops::v4<uint8_t>, int2>(state); })
                    ->Unit(benchmark::kMicrosecond)
                    ->UseRealTime()
                    ->Args({morph_op, kernel_size, se_shape});
            }
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        // 初始化测试数据（使用默认参数）
        std::cout << "Init benchmark test data..." << std::endl;
        InitializeBenchmarkData("D:/Project/dianjiao/2025_08_18/picture_1_2025_08_18_18_02_25_059.png");
        std::cout << "Init Done!" << std::endl;

        // 运行 benchmark
        std::cout << "Run benchmark..." << std::endl;
        std::cout << "Config: \n"
                  << "\t Image size: " << g_benchmark_data->test_image.size << std::endl;
        std::cout << "\t Testing combinations:" << std::endl;
        std::cout << "\t - Operations: " << morph_ops.size() << " types" << std::endl;
        std::cout << "\t - Shapes: " << se_shapes.size() << " types" << std::endl;
        std::cout << "\t - Kernel sizes: " << kernel_sizes.size() << " sizes" << std::endl;
        std::cout << "\t - Total combinations: "
                  << morph_ops.size() * se_shapes.size() * kernel_sizes.size() * 6 // 5个CUDA版本 + 1个OpenCV版本
                  << std::endl;
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