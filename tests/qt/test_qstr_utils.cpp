#include "../values_test.h"
#include "qstr_utils.h"

#include <gtest/gtest.h>

using namespace free_kick::utils::qstr;

// clang-format off

FREE_KICK_TEST_SUITE_P(Vec2QStrTest, ValueList<const std::vector<double>&, const std::string&>
{
    // input, output
    {{1.,2.,3.}, "1, 2, 3, "},
    {{-1.,-2.,-3.}, "-1, -2, -3, "},
    {{0.,0.,0.}, "0, 0, 0, "},
    {{0.5,0.6,0.7}, "0.5, 0.6, 0.7, "},
    {{0.1,0.2,0.3,0.4,0.5,0.6,0.7}, "0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, "},
    {{0.11,0.22,0.33,0.44,0.55,0.66,0.77}, "0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, "},
});

// clang-format on

TEST_P(Vec2QStrTest, ToQStr)
{
    EXPECT_EQ(toQString(GetParamValue<0>()).toStdString(), GetParamValue<1>());
}

TEST_P(Vec2QStrTest, ToQStrv2)
{
    EXPECT_EQ(toQStringv2(GetParamValue<0>()).toStdString(), GetParamValue<1>());
}
