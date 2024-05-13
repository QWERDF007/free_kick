#include "test.h"

#include <iostream>

void func1(int x)
{
    std::cout << __FUNCTION__ << " " << __LINE__ << std::endl;
    func2(x);
    std::cout << __FUNCTION__ << " " << __LINE__ << std::endl;
}

void func2(int x)
{
    std::cout << __FUNCTION__ << " " << __LINE__ << std::endl;
    test();
    std::cout << __FUNCTION__ << " " << __LINE__ << std::endl;
}

void test()
{
    std::cout << __FUNCTION__ << " " << __LINE__ << std::endl;
    char *p = (char *)0xadef1234;
    *p      = 0; // crash here
    int a   = 1;
    int b   = 0;
    int c   = a / b; // divide by zero error
    std::cout << __FUNCTION__ << " " << __LINE__ << std::endl;
}