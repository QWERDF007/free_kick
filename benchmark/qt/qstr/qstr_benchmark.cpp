#include "qstr_utils.h"

#include <benchmark/benchmark.h>

#include <algorithm>

using namespace free_kick::utils::qstr;

static void BM_Vec2QString(benchmark::State &state)
{
    std::vector<double> vec(100);
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
    state.counters["Foo"] = 42;
    state.counters["Bar"] = 100;
}

// Register the function as a benchmark
BENCHMARK(BM_Vec2QString);

BENCHMARK_MAIN();
