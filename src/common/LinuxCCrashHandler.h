#pragma once

#include <string>

namespace free_kick::common {

class LinuxCCrashHandler
{
public:
    explicit LinuxCCrashHandler() = default;
    ~LinuxCCrashHandler()         = default;

    void setup();

private:
#if defined(__linux__)
    static std::string GetExceptionName(int code);
    static std::string GetExceptionModule(const pid_t pid);

    static std::string GetCurrentTraceBackString(const int frame_to_skip = 0);

    static void CreateMiniDump(const std::string &filename, const std::string &msg);

    static std::string HandleCommonException(int);

    static void SingalHandler(int);
#endif
};

} // namespace free_kick::common