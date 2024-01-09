#include "../values_test.h"
#include "algo_utils.h"

#include <gtest/gtest.h>

#include <vector>

// clang-format off

std::vector<int> arr1{1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
std::vector<int> arr2{1, 1, 1, 2, 2, 4, 3, 3, 4, 4, 4, 4, 5, 7, 0};
std::vector<int> arr3{1, 1, 2, 2, 3, 3, 4, 4};
std::vector<int> arr4;

FREE_KICK_TEST_SUITE_P(MostFrequentElementTest, ValueList<const std::vector<int>&, int, int>
{
    // arr, element, freq
    {arr1, 4, 4},
    {arr2, 4, 5},
    {arr3, 1, 2},
    {arr4, std::numeric_limits<int>::min(), 0},
});

// clang-format on

TEST_P(MostFrequentElementTest, most_frequent_element)
{
    auto arr     = GetParamValue<0>();
    auto element = GetParamValue<1>();
    auto freq    = GetParamValue<2>();
    int  found_element;
    int  found_freq;
    free_kick::utils::findMostFrequentElement(arr, found_element, found_freq);
    EXPECT_EQ(found_element, element);
    EXPECT_EQ(found_freq, freq);
};