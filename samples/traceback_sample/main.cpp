#include "CrashHandler.h"
#include "test.h"

#include <exception>
#include <iostream>

void fuc()
{
    char *p = (char *)0xdefa1234;
    *p      = 0;
}

int main(int argc, char *argv[])
{
    auto crash_handler = free_kick::common::CrashHandler();
    crash_handler.setup();
    fuc();
    func1(42);

    return 0;
}
