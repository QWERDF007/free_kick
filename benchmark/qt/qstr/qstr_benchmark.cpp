#include "qstr_utils.h"

#include <benchmark/benchmark.h>

#include <algorithm>

using namespace free_kick::utils::qstr;

static void BM_Vec2QString(benchmark::State &state)
{
    std::vector<double> vec(state.range(0));
    // 使用std::generate填充vector
    std::generate(vec.begin(), vec.end(),
                  []()
                  {
                      static int i = 1;
                      return static_cast<double>(i++);
                  });
    for (auto _ : state)
    {
        toQString(vec);
    }
}

static void BM_Vec2QStringv2(benchmark::State &state)
{
    std::vector<double> vec(state.range(0));
    // 使用std::generate填充vector
    std::generate(vec.begin(), vec.end(),
                  []()
                  {
                      static int i = 1;
                      return static_cast<double>(i++);
                  });
    for (auto _ : state)
    {
        toQStringv2(vec);
    }
}

// Register the function as a benchmark
BENCHMARK(BM_Vec2QString)->RangeMultiplier(10)->Range(1e1, 1e4)->Repetitions(10);
BENCHMARK(BM_Vec2QStringv2)->RangeMultiplier(10)->Range(1e1, 1e4)->Repetitions(10);

BENCHMARK_MAIN();
