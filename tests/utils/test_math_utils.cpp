#include "../values_test.h"
#include "math_utils.h"

#include <gtest/gtest.h>

// clang-format off

FREE_KICK_TEST_SUITE_P(MathRandomArrayTest, ValueList<size_t, int, int, double, double>
{
    // total,  min,    max, min,    max,
    {      0,    0,     0,   0.,     0.,},
    {      5,    0,    10,   0.,    10.,},
    {     50,    0,    10,   0.,    10.,},
    {     50,   11, 10125,  15., 12530.,},
});

// clang-format on

TEST_P(MathRandomArrayTest, random_array)
{
    const size_t total      = GetParamValue<0>();
    const int    int_min    = GetParamValue<1>();
    const int    int_max    = GetParamValue<2>();
    const double double_min = GetParamValue<3>();
    const double double_max = GetParamValue<4>();

    auto int_data    = free_kick::utils::random<int>(int_min, int_max, total);
    auto double_data = free_kick::utils::random<double>(double_min, double_max, total);
    EXPECT_EQ(total, int_data.size());
    EXPECT_EQ(total, double_data.size());
    if (total > 0)
    {
        auto int_min_value = std::min_element(int_data.begin(), int_data.end());
        auto int_max_value = std::max_element(int_data.begin(), int_data.end());
        EXPECT_GE(*int_min_value, int_min);
        EXPECT_LE(*int_max_value, int_max);

        auto double_min_value = std::min_element(int_data.begin(), int_data.end());
        auto double_max_value = std::max_element(int_data.begin(), int_data.end());
        EXPECT_GE(*double_min_value, double_min);
        EXPECT_LE(*double_max_value, double_max);
    }
}