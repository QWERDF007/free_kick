#include "LinuxCCrashHandler.h"

#if defined(__linux__)
#    include <cxxabi.h> // __cxa_demangle
#    include <dlfcn.h>  // dladdr
#    include <execinfo.h>
#    include <link.h> // link_map
#    include <signal.h>
#    include <unistd.h> // sleep, usleep
#endif

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#ifdef __GNUC__
/**
 * @brief 将函数名称修饰去除，返回原有的函数名称
*/
inline std::string demangle(std::string name)
{
    int status = -4; // some arbitrary value to eliminate the compiler warning

    std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(name.c_str(), NULL, NULL, &status), std::free};
    return (status == 0) ? res.get() : name;
}
#else
inline std::string demangle(std::string name)
{
    return name;
}
#endif

namespace free_kick::common {

// clang-format off
#if defined(__linux__)

#define EX_CASE(code) \
    case code:        \
        return #code;

std::string LinuxCCrashHandler::GetExceptionName(int code)
{
    switch (code)
    {
        EX_CASE(SIGINT);
        EX_CASE(SIGILL);
        EX_CASE(SIGABRT);
        EX_CASE(SIGFPE);
        EX_CASE(SIGSEGV);
        EX_CASE(SIGTERM);

        EX_CASE(SIGHUP);
        EX_CASE(SIGQUIT);
        EX_CASE(SIGTRAP);
        EX_CASE(SIGKILL);
        EX_CASE(SIGBUS);
        EX_CASE(SIGSYS);
        EX_CASE(SIGPIPE);
        EX_CASE(SIGALRM);

        EX_CASE(SIGURG);
        EX_CASE(SIGSTOP);
        EX_CASE(SIGTSTP);
        EX_CASE(SIGCONT);
        EX_CASE(SIGCHLD);
        EX_CASE(SIGTTIN);
        EX_CASE(SIGTTOU);
        EX_CASE(SIGPOLL);
        EX_CASE(SIGXCPU);
        EX_CASE(SIGXFSZ);
        EX_CASE(SIGVTALRM);
        EX_CASE(SIGPROF);
        EX_CASE(SIGUSR1);
        EX_CASE(SIGUSR2);
    default:
        return "Unknown exception";
    }
}

#undef EX_CASE

// clang-format on

std::string LinuxCCrashHandler::GetExceptionModule(const pid_t pid)
{
    int  len = 1024;
    char sztmp[32];
    char path[len];
    sprintf(sztmp, "/proc/%d/exe", pid);
    int bytes = readlink(sztmp, path, len);
    if (bytes < len - 1)
        bytes = len - 1;
    if (bytes >= 0)
        path[bytes] = '\0';
    return path;
}

size_t ConvertToVMA(size_t addr)
{
    Dl_info   info;
    link_map *link_map;
    dladdr1((void *)addr, &info, (void **)&link_map, RTLD_DL_LINKMAP);
    return addr - link_map->l_addr;
}

std::string LinuxCCrashHandler::GetCurrentTraceBackString(const int frame_to_skip)
{
    std::ostringstream sout;
    sout << "\n\n--------------------------------------\n";
    sout << "C++ Traceback (most recent call last):";
    sout << "\n--------------------------------------\n";

    static constexpr int TRACE_STACK_LIMIT = 100;

    void *call_stack[TRACE_STACK_LIMIT];
    auto  size    = backtrace(call_stack, TRACE_STACK_LIMIT);
    auto  symbols = backtrace_symbols(call_stack, size);

    for (int i = size - 1; i >= frame_to_skip; --i)
    {
        Dl_info   info;
        link_map *link_map;
        if (dladdr1(call_stack[i], &info, (void **)&link_map, RTLD_DL_LINKMAP))
        {
            // use addr2line; dladdr itself is rarely useful (see doc)

            // size_t VMA_addr = ConvertToVMA((size_t)call_stack[i]);
            size_t VMA_addr = (size_t)call_stack[i] - link_map->l_addr;
            // https://stackoverflow.com/questions/11579509/wrong-line-numbers-from-addr2line/63841497#63841497
            //if(i!=crash_depth)
            VMA_addr -= 1;
#    ifdef NDEBUG
            sout << symbols[i] << std::endl;
            sout << "    info.dli_fname: " << info.dli_fname;
            if (info.dli_sname)
                sout << " info.dli_sname: " << info.dli_sname;
            sout << " VMA_addr: " << std::hex << VMA_addr << std::endl;

#    else
            char command[256];
            snprintf(command, sizeof(command), "addr2line -Cif -p -e %s  %zx > tmp.txt", info.dli_fname, VMA_addr);
            system(command);
            std::ifstream file("tmp.txt");
            std::string   line;
            while (std::getline(file, line))
            {
                sout << line << std::endl;
            }
            file.close();
            remove("tmp.txt");
#    endif
        }
    }
    free(symbols);

    return sout.str();
}

void LinuxCCrashHandler::CreateMiniDump(const std::string &filename, const std::string &msg)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open())
        return;
    ofs << msg;
    ofs.close();
}

std::string LinuxCCrashHandler::HandleCommonException(int code)
{
    std::ostringstream sout;

    std::string module_name    = GetExceptionModule(getpid());
    std::string exception_name = GetExceptionName(code);

    sout << "An exception has occured which was not handled!\n"
         << "Code: " << exception_name << "(" << code << ")\nModule: " << module_name << std::endl;

    auto traceback_msg = GetCurrentTraceBackString(4);
    sout << traceback_msg;

    CreateMiniDump("crashdump.dmp", traceback_msg);

    return sout.str();
}

void LinuxCCrashHandler::SingalHandler(int signal)
{
    auto exception_msg = HandleCommonException(signal);
    std::cerr << exception_msg << std::endl;
    if (crash_callback)
        crash_callback();
    exit(signal);
}

#endif // __linux__

std::function<void()> LinuxCCrashHandler::crash_callback = nullptr;

void LinuxCCrashHandler::setup(std::function<void()> func)
{
    crash_callback = func;
#if defined(__linux__)
    signal(SIGINT, SingalHandler); // Ctrl + C
    signal(SIGILL, SingalHandler);
    signal(SIGABRT, SingalHandler);
    signal(SIGFPE, SingalHandler);
    signal(SIGQUIT, SingalHandler); // Ctrl + Break
    signal(SIGKILL, SingalHandler); // kill -9 (cannot be caught or ignored)
    signal(SIGTSTP, SingalHandler); // Ctrl + Z (cannot be caught or ignored)
    signal(SIGSEGV, SingalHandler); // Invalid access to storage.
    signal(SIGTERM, SingalHandler); // kill -15
    signal(SIGHUP, SingalHandler);  // Hangup
#endif
}

} // namespace free_kick::common