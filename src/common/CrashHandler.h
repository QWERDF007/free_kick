#pragma once
#include <functional>

namespace free_kick::common {

class CrashHandler
{
public:
    explicit CrashHandler() = default;
    void setup(std::function<void()> crash_callback = nullptr);

private:
};

} // namespace free_kick::common
