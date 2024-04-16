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
});

// clang-format on

TEST_P(Vec2QStrTest, ToQStr)
{
    EXPECT_EQ(toQString(GetParamValue<0>()).toStdString(), GetParamValue<1>());
}
