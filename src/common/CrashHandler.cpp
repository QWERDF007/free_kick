#include "CrashHandler.h"

#if defined(_WIN32)
#    include "WindowsCCrashHandler.h"
#else

#endif

namespace free_kick::common {

void CrashHandler::setup()
{
#if defined(_WIN32)
    WindowsCCrashHandler ccrash_handler;
    ccrash_handler.setup();
#else

#endif
}

} // namespace free_kick::common