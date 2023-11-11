#include "../values_test.h"
#include "math_utils.h"

#include <gtest/gtest.h>

// clang-format off

FREE_KICK_TEST_SUITE_P(MathUtilsTest, ValueList<int, int, int>
{
    // total, min, max, 
    {      0,   0,   0, },
    {      5,   0,   10, },
});

// clang-format on

TEST_P(MathUtilsTest, random)
{
    const int total = GetParamValue<0>();
    const int min   = GetParamValue<1>();
    const int max   = GetParamValue<2>();

    auto data = free_kick::utils::random<int>(min, max, total);
    EXPECT_EQ(total, data.size());
    if (total > 0)
    {
        auto max_value = std::max_element(data.begin(), data.end());
        auto min_value = std::min_element(data.begin(), data.end());
        EXPECT_LE(*max_value, max);
        EXPECT_GE(*min_value, min);
    }
}