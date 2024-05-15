#pragma once

// https://www.codeproject.com/Articles/207464/Exception-Handling-in-Visual-Cplusplus

#include <string>

#if defined(_WIN32)
#    include <Windows.h>
#endif

namespace free_kick::common {

class WindowsCCrashHandler
{
public:
    explicit WindowsCCrashHandler() = default;
    ~WindowsCCrashHandler()         = default;

    void setup();

private:
#if defined(_WIN32)

    /**
     * @brief 获取异常名称
     * @param code 
     * @return LPCSTR 
     */
    static std::string GetExceptionName(DWORD code);

    /**
     * @brief 获取异常对象指针
     * @param dwExceptionCode 异常代码
     * @param ppExceptionPointers 异常对象指针
     */
    static void GetExceptionPointers(DWORD dwExceptionCode, EXCEPTION_POINTERS **ppExceptionPointers);

    /**
     * @brief 获取异常发生的模块名称
     * @param process 进程句柄
     * @param address 异常发生的地址
     * @param module_name 保存模块名称的缓冲区
     * @return HMODULE 异常发生的模块句柄
     */
    static std::string GetExceptionModule(HANDLE process, LPVOID address);

    /**
     * @brief 通过 `MiniDumpWriteDump` 将用户模式小型转储信息写入指定的文件 crashdump.dmp
     * @param exception 异常信息指针
     */
    static void CreateMiniDump(const std::string &filename, EXCEPTION_POINTERS *exception, const std::string &msg);

    /**
     * @brief 打印调用栈，文件名，行号，函数名称
     * @note 只在 Debug 模式下生效
     * @param process 进程句柄
     */
    static std::string GetCurrentTraceBackString(HANDLE process, const ULONG frames_to_skip = 0);

    /**
     * @brief 处理内存访问违例异常，打印类型，模块名称，线程ID，地址，偏移，代码基
     * @param process 进程句柄
     * @param exception 异常信息指针
     * @param frames_to_skip backtrace 栈帧跳过数量
     */
    static void HandleAccessViolation(HANDLE process, LPEXCEPTION_POINTERS exception, const ULONG frames_to_skip = 0);

    /**
     * @brief 处理常规异常，打印异常名称，模块名称
     * @param process 进程句柄
     * @param exception 异常信息指针
     * @param frames_to_skip backtrace 栈帧跳过数量
     */
    static void HandleCommonException(HANDLE process, LPEXCEPTION_POINTERS exception, const ULONG frames_to_skip = 0);

    // __stdcall: https://learn.microsoft.com/zh-CN/cpp/cpp/stdcall?view=msvc-160
    // __cdecl: https://learn.microsoft.com/zh-cn/cpp/cpp/cdecl?view=msvc-160

    /**
     * @brief 顶级异常处理函数，捕获异常调用栈，创建内存转储文件
     * @param exception 异常信息指针
     * @return LONG 异常处理结果
     */
    static LONG WINAPI UnhandledExceptionHandler(LPEXCEPTION_POINTERS exception);

    // CRT terminate() call handler
    static void __cdecl TerminateHandler();

    // CRT unexpected() call handler
    static void __cdecl UnexpectedHandler();

    // CRT Pure virtual method call handler
    static void __cdecl PureCallHandler();

    // CRT invalid parameter handler
    static void __cdecl InvalidParameterHandler(const wchar_t *expression, const wchar_t *function, const wchar_t *file,
                                                unsigned int line, uintptr_t pReserved);

    // CRT new operator fault handler
    static int __cdecl NewHandler(size_t);

    // CRT SIGABRT signal handler
    static void SIGABRTHandler(int);

    // CRT SIGFPE signal handler
    static void SIGFPEHandler(int, int);

    // CRT SIGINT signal handler
    static void SIGINTHandler(int);

    // CRT SIGILL signal handler
    static void SIGILLHandler(int);

    // CRT SIGSEGV signal handler
    static void SIGSEGVHandler(int);

    // CRT SIGTERM signal handler
    static void SIGTERMHandler(int);

    void SetProcessExceptionHandlder();
    void SetThreadExceptionHandlder();

#endif
};

} // namespace free_kick::common
