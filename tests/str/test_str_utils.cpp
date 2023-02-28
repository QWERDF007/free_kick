#include <gtest/gtest.h>
#include "str/str_utils.h"


TEST(StrTest, strip_test_space)
{
    ASSERT_EQ(strip(" abcd"), "abcd");
    ASSERT_EQ(strip(" abcd "), "abcd");
    ASSERT_EQ(strip("  abcd "), "abcd");
    ASSERT_EQ(strip("  ab cd "), "ab cd");
    ASSERT_EQ(strip("  ab  cd "), "ab  cd");
    ASSERT_EQ(strip("      a b  c d     "), "a b  c d");

    ASSERT_NE(strip(" abcd"), " abcd");
    ASSERT_NE(strip("abcd   "), "abcd   ");
    ASSERT_NE(strip("   abcd   "), "   abcd   ");
    ASSERT_NE(strip(" abcd "), " abcd ");
    ASSERT_NE(strip(" ab cd "), "ab  cd");
}

TEST(StrTest, strip_test_tab)
{
    ASSERT_EQ(strip("\tabcd"), "abcd");
    ASSERT_EQ(strip("\tabcd\t"), "abcd");
    ASSERT_EQ(strip("\ta\tbc\td\t"), "a\tbc\td");
    ASSERT_EQ(strip("\t\tabcd\t\t\t"), "abcd");

    ASSERT_NE(strip("\tabcd"), "\tabcd");
    ASSERT_NE(strip("\tabcd\t"), "\tabcd\t");
    ASSERT_NE(strip("\t\tabcd\t\t\t"), "\t\tabcd\t\t\t");
}

TEST(StrTest, strip_test_newline)
{
    ASSERT_EQ(strip("\nabcd"), "abcd");
    ASSERT_EQ(strip("abcd\n"), "abcd");
    ASSERT_EQ(strip("ab\ncd"), "ab\ncd");
    ASSERT_EQ(strip("\n\nabcd\n\n"), "abcd");
    ASSERT_EQ(strip("\n\nab cd\n\n"), "ab cd");

    ASSERT_NE(strip("\nab cd"), "\nab cd");
    ASSERT_NE(strip("ab cd\n"), "ab cd\n");
    ASSERT_NE(strip("\nab cd\n"), "\nab cd\n");
}