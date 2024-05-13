#include "CrashHandler.h"

#if defined(_WIN32)
#    include "WindowsCrashHandler.h"
#else

#endif

namespace free_kick::common {

void CrashHandler::setup()
{
#if defined(_WIN32)
    CCrashHandler ccrash_handler;
    ccrash_handler.setup();
#else

#endif
}

} // namespace free_kick::common