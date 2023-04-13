#include <iostream>

#include "utils/str_utils.h"

int main(int argc, char **argv)
{
    std::cout << free_kick::strip("abcd ") << std::endl;;
    return 0;
}