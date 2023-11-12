#include "../values_test.h"
#include "str_utils.h"

#include <gtest/gtest.h>

// using free_kick::strip;
using free_kick::utils::strip;

// clang-format off

FREE_KICK_TEST_SUITE_P(StripEQTest, ValueList<std::string &, std::string>
{
//                   input,     result
    {              " abcd",     "abcd"},
    {             " abcd ",     "abcd"},
    {            "  abcd ",     "abcd"},
    {           "  ab cd ",    "ab cd"},
    {          "  ab  cd ",   "ab  cd"},
    {"      a b  c d     ", "a b  c d"},

    {             "\tabcd",     "abcd"},
    {     "\t\tabcd\t\t\t",     "abcd"},

    {             "\nabcd",     "abcd"},
    {      "\n\nab cd\n\n",    "ab cd"},
});

FREE_KICK_TEST_SUITE_P(StripNEQTest, ValueList<std::string &, std::string>
{
//           input,       result
    {      " abcd",      " abcd"},
    {    "abcd   ",    "abcd   "},
    { "   abcd   ", "   abcd   "},
    {     " abcd ",     " abcd "},
    {    " ab cd ",     "ab cd "},

    {     "\tabcd",     "\tabcd"},
    {   "\tabcd\t",   "\tabcd\t"},

    {    "\nab cd",    "\nab cd"},
    {  "\nab cd\n",  "\nab cd\n"},
});

// clang-format on

TEST_P(StripEQTest, strip_test_eq)
{
    EXPECT_EQ(strip(GetParamValue<0>()), GetParamValue<1>());
}

TEST_P(StripNEQTest, strip_test_neq)
{
    EXPECT_NE(strip(GetParamValue<0>()), GetParamValue<1>());
}
