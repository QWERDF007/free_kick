#include "CrashHandler.h"

#if defined(_WIN32)
#    include "WindowsCCrashHandler.h"
#else
#    include "LinuxCCrashHandler.h"
#endif

namespace free_kick::common {

void CrashHandler::setup(std::function<void()> crash_callback)
{
#if defined(_WIN32)
    WindowsCCrashHandler ccrash_handler;
#else
    LinuxCCrashHandler ccrash_handler;
#endif
    ccrash_handler.setup(crash_callback);
}

} // namespace free_kick::common