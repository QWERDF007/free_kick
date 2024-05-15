#include "WindowsCCrashHandler.h"

#if defined(_WIN32)
#    include <DbgHelp.h> // for SymGetLineFromAddr etc.
#    include <new.h>     // for _set_new_handler
#    include <psapi.h>   // for EnumProcessModules
#    include <signal.h>  // for signal etc.
#endif

#include <fstream>
#include <iostream>
#include <sstream>

#ifndef _AddressOfReturnAddress

// Taken from: http://msdn.microsoft.com/en-us/library/s975zw7k(VS.71).aspx
#    ifdef __cplusplus
#        define EXTERNC extern "C"
#    else
#        define EXTERNC
#    endif
// _ReturnAddress and _AddressOfReturnAddress should be prototyped before use
EXTERNC void *_AddressOfReturnAddress(void);
EXTERNC void *_ReturnAddress(void);
#endif

namespace free_kick::common {

// clang-format off

#if defined(_WIN32)

// https://win32easy.blogspot.com/2011/03/exception-handling-inform-your-users_26.html


#define EX_CASE(code) \
    case code:        \
        return #code;

std::string WindowsCCrashHandler::GetExceptionName(DWORD code)
{
    switch (code)
    {
        EX_CASE(EXCEPTION_ACCESS_VIOLATION);
        EX_CASE(EXCEPTION_DATATYPE_MISALIGNMENT);
        EX_CASE(EXCEPTION_BREAKPOINT);
        EX_CASE(EXCEPTION_SINGLE_STEP);
        EX_CASE(EXCEPTION_ARRAY_BOUNDS_EXCEEDED);
        EX_CASE(EXCEPTION_FLT_DENORMAL_OPERAND);
        EX_CASE(EXCEPTION_FLT_DIVIDE_BY_ZERO);
        EX_CASE(EXCEPTION_FLT_INEXACT_RESULT);
        EX_CASE(EXCEPTION_FLT_INVALID_OPERATION);
        EX_CASE(EXCEPTION_FLT_OVERFLOW);
        EX_CASE(EXCEPTION_FLT_STACK_CHECK);
        EX_CASE(EXCEPTION_FLT_UNDERFLOW);
        EX_CASE(EXCEPTION_INT_DIVIDE_BY_ZERO);
        EX_CASE(EXCEPTION_INT_OVERFLOW);
        EX_CASE(EXCEPTION_PRIV_INSTRUCTION);
        EX_CASE(EXCEPTION_IN_PAGE_ERROR);
        EX_CASE(EXCEPTION_ILLEGAL_INSTRUCTION);
        EX_CASE(EXCEPTION_NONCONTINUABLE_EXCEPTION);
        EX_CASE(EXCEPTION_STACK_OVERFLOW);
        EX_CASE(EXCEPTION_INVALID_DISPOSITION);
        EX_CASE(EXCEPTION_GUARD_PAGE);
        EX_CASE(EXCEPTION_INVALID_HANDLE);

    case 0xE06D7363:
        return "C++ Exception";

    default:
        return "Unknown exception";
    }
}

#undef EX_CASE

void WindowsCCrashHandler::GetExceptionPointers(DWORD dwExceptionCode, EXCEPTION_POINTERS **ppExceptionPointers)
{
	// The following code was taken from VC++ 8.0 CRT (invarg.c: line 104)
	EXCEPTION_RECORD ExceptionRecord;
	CONTEXT ContextRecord;
	memset(&ContextRecord, 0, sizeof(CONTEXT));

#ifdef _X86_

	__asm {
		mov dword ptr [ContextRecord.Eax], eax
			mov dword ptr [ContextRecord.Ecx], ecx
			mov dword ptr [ContextRecord.Edx], edx
			mov dword ptr [ContextRecord.Ebx], ebx
			mov dword ptr [ContextRecord.Esi], esi
			mov dword ptr [ContextRecord.Edi], edi
			mov word ptr [ContextRecord.SegSs], ss
			mov word ptr [ContextRecord.SegCs], cs
			mov word ptr [ContextRecord.SegDs], ds
			mov word ptr [ContextRecord.SegEs], es
			mov word ptr [ContextRecord.SegFs], fs
			mov word ptr [ContextRecord.SegGs], gs
			pushfd
			pop [ContextRecord.EFlags]
	}

	ContextRecord.ContextFlags = CONTEXT_CONTROL;
#pragma warning(push)
#pragma warning(disable:4311)
	ContextRecord.Eip = (ULONG)_ReturnAddress();
	ContextRecord.Esp = (ULONG)_AddressOfReturnAddress();
#pragma warning(pop)
	ContextRecord.Ebp = *((ULONG *)_AddressOfReturnAddress()-1);


#elif defined (_IA64_) || defined (_AMD64_)

	/* Need to fill up the Context in IA64 and AMD64. */
	RtlCaptureContext(&ContextRecord);

#else  /* defined (_IA64_) || defined (_AMD64_) */

	ZeroMemory(&ContextRecord, sizeof(ContextRecord));

#endif  /* defined (_IA64_) || defined (_AMD64_) */

	ZeroMemory(&ExceptionRecord, sizeof(EXCEPTION_RECORD));

	ExceptionRecord.ExceptionCode = dwExceptionCode;
	ExceptionRecord.ExceptionAddress = _ReturnAddress();

	///

	EXCEPTION_RECORD* pExceptionRecord = new EXCEPTION_RECORD;
	memcpy(pExceptionRecord, &ExceptionRecord, sizeof(EXCEPTION_RECORD));
	CONTEXT* pContextRecord = new CONTEXT;
	memcpy(pContextRecord, &ContextRecord, sizeof(CONTEXT));

	*ppExceptionPointers = new EXCEPTION_POINTERS;
	(*ppExceptionPointers)->ExceptionRecord = pExceptionRecord;
    (*ppExceptionPointers)->ContextRecord = pContextRecord;
}

// clang-format on

std::string wcharToString(const wchar_t *wstr)
{
    if (wstr == nullptr)
        return "";
    // 首先计算需要的字符数量
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);

    // 创建一个足够大的字符数组来存储转换后的字符串
    std::string str(len, 0);

    // 执行转换
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, &str[0], len, nullptr, nullptr);

    return str;
}

std::wstring stringToWchar(const std::string &str)
{
    if (str.empty())
        return L"";

    // 首先计算需要的宽字符数量
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);

    // 创建一个足够大的宽字符数组来存储转换后的字符串
    std::wstring wstr(len, 0);

    // 执行转换
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], len);

    return wstr;
}

std::string WindowsCCrashHandler::GetExceptionModule(HANDLE process, LPVOID address)
{
    HMODULE module_list[1024];
    DWORD   size_needed = 0;
    if (FALSE == EnumProcessModules(process, module_list, 1024, &size_needed) || size_needed < sizeof(HMODULE))
        return NULL;

    int cur_module = -1;

    int size = (size_needed / sizeof(HMODULE));
    for (int i = 0; i < size; ++i)
    {
        if ((DWORD)module_list[i] < (DWORD)address)
        {
            if (cur_module == -1)
                cur_module = i;
            else if ((DWORD)module_list[cur_module] < (DWORD)module_list[i])
                cur_module = i;
        }
    }
    if (cur_module == -1)
        return "Unknown module";

#    ifdef UNICODE
    WCHAR module_name[MAX_PATH];
#    else
    CHAR module_name[MAX_PATH];
#    endif
    if (!GetModuleFileName(module_list[cur_module], module_name, MAX_PATH))
        return "Unknown module";
#    ifdef UNICODE
    return wcharToString(module_name);
#    else
    return module_name;
#    endif
}

std::string WindowsCCrashHandler::GetCurrentTraceBackString(HANDLE process, const ULONG frames_to_skip)
{
    static constexpr int TRACE_STACK_LIMIT = 128;

    SymInitialize(process, NULL, TRUE);

    void *stack_trace[TRACE_STACK_LIMIT];
    ULONG frames_captured = CaptureStackBackTrace(frames_to_skip, TRACE_STACK_LIMIT, stack_trace, NULL);

    SYMBOL_INFO  *symbol;
    DWORD         sym_options;
    IMAGEHLP_LINE line = {sizeof(IMAGEHLP_LINE)};

    // 获取当前进程的符号选项
    sym_options = SymGetOptions();
    // 设置符号选项，确保获取源文件信息
    sym_options |= SYMOPT_LOAD_LINES;
    sym_options |= SYMOPT_DEBUG;
    SymSetOptions(sym_options);

    std::ostringstream sout;
    sout << "\n\n--------------------------------------\n";
    sout << "C++ Traceback (most recent call last):";
    sout << "\n--------------------------------------\n";

    // std::cerr << "\nTraceback (most recent call last): " << std::endl;
    constexpr int end_idx = 0; // 0: PrintStackTrace
    for (int i = frames_captured - 1; i >= end_idx; --i)
    {
        symbol = (SYMBOL_INFO *)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(TCHAR), 1);

        symbol->MaxNameLen   = 255;
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

        DWORD displacement = 0;
        bool  found_line   = SymGetLineFromAddr(process, (DWORD64)(stack_trace[i]), &displacement, &line);
        bool  found_symbol = SymFromAddr(process, (DWORD64)(stack_trace[i]), 0, symbol);
        if (found_line && found_symbol)
            sout << "File \"" << line.FileName << "\", line " << line.LineNumber << ", in " << symbol->Name
                 << std::endl;
        else if (found_line)
            sout << "File \"" << line.FileName << "\", line " << line.LineNumber << std::endl;
        // 清理符号缓存
        free(symbol);
    }
    SymCleanup(process);
    return sout.str();
}

void WindowsCCrashHandler::CreateMiniDump(const std::string &filename, EXCEPTION_POINTERS *exception,
                                          const std::string &msg)
{
    std::ofstream ofs(filename + ".txt");
    if (ofs.is_open())
    {
        ofs << msg << std::endl;
        ofs.close();
    }

    HANDLE hFile = NULL;
    // Create the minidump file
#    ifdef UNICODE
    std::wstring w_filename = stringToWchar(filename);
    hFile = CreateFile(w_filename.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
#    else
    hFile = CreateFile(filename.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
#    endif
    // Couldn't create file
    if (hFile == INVALID_HANDLE_VALUE)
        return;

    MINIDUMP_EXCEPTION_INFORMATION ExceptionParam;
    ExceptionParam.ThreadId          = GetCurrentThreadId();
    ExceptionParam.ExceptionPointers = exception;
    ExceptionParam.ClientPointers    = FALSE;

    // Write minidump to the file
    bool bWriteDump = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpNormal,
                                        &ExceptionParam, NULL, NULL);
    // Error writing dump.
    if (!bWriteDump)
        return;

    // Close file
    CloseHandle(hFile);
}

void WindowsCCrashHandler::HandleAccessViolation(HANDLE process, LPEXCEPTION_POINTERS exception,
                                                 const ULONG frames_to_skip)
{
    std::string module_name = GetExceptionModule(process, exception->ExceptionRecord->ExceptionAddress);

    DWORD code_base = (DWORD)GetModuleHandle(NULL);
    DWORD offset    = (DWORD)exception->ExceptionRecord->ExceptionAddress - code_base;

    std::string access_type;
    switch (exception->ExceptionRecord->ExceptionInformation[0])
    {
    case 0:
        access_type = "Read";
        break;
    case 1:
        access_type = "Write";
        break;
    case 2:
        access_type = "Execute";
        break;
    default:
        access_type = "Unknown";
        break;
    }

    std::ostringstream sout;
    sout << "An exception has occured which was not handled!\n"
         << "Code: " << GetExceptionName(exception->ExceptionRecord->ExceptionCode) << "("
         << exception->ExceptionRecord->ExceptionCode << ")\n"
         << "Module: " << module_name << "\n"
         << "The thread " << GetCurrentThreadId() << " tried to " << access_type << " memory at address 0x" << std::hex
         << exception->ExceptionRecord->ExceptionInformation[1] << " which is inaccessible!\n"
         << "Offset: 0x" << std::hex << offset << "\n"
         << "Codebase: 0x" << std::hex << code_base << std::endl;
#    ifndef NDEBUG
    sout << GetCurrentTraceBackString(process, frames_to_skip);
#    endif
    auto msg = sout.str();
    CreateMiniDump("crashdump.dmp", exception, msg);
    std::cerr << msg << std::endl;
}

void WindowsCCrashHandler::HandleCommonException(HANDLE process, LPEXCEPTION_POINTERS exception,
                                                 const ULONG frames_to_skip)
{
    std::string module_name = GetExceptionModule(process, exception->ExceptionRecord->ExceptionAddress);

    std::ostringstream sout;
    sout << "An exception has occured which was not handled!\n"
         << "Code: " << GetExceptionName(exception->ExceptionRecord->ExceptionCode) << "("
         << exception->ExceptionRecord->ExceptionCode << ")\n"
         << "Module: " << module_name << std::endl;
#    ifndef NDEBUG
    sout << GetCurrentTraceBackString(process, frames_to_skip);
#    endif
    auto msg = sout.str();
    CreateMiniDump("crashdump.dmp", exception, msg);
    std::cerr << msg << std::endl;
}

LONG WINAPI WindowsCCrashHandler::UnhandledExceptionHandler(LPEXCEPTION_POINTERS exception)
{
    HANDLE process = GetCurrentProcess();

    switch (exception->ExceptionRecord->ExceptionCode)
    {
    case EXCEPTION_ACCESS_VIOLATION:
        HandleAccessViolation(process, exception, 4);
        break;
    default:
        HandleCommonException(process, exception, 4);
        break;
    }

    TerminateProcess(GetCurrentProcess(), 1);

    return EXCEPTION_EXECUTE_HANDLER;
}

void __cdecl WindowsCCrashHandler::TerminateHandler()
{
    // Abnormal program termination (terminate() function was called)

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    TerminateProcess(GetCurrentProcess(), 1);
}

void __cdecl WindowsCCrashHandler::UnexpectedHandler()
{
    // Unexpected error (unexpected() function was called)

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    TerminateProcess(GetCurrentProcess(), 1);
}

void WindowsCCrashHandler::PureCallHandler()
{
    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    TerminateProcess(GetCurrentProcess(), 1);
}

#    define UNUSED(x) (void)(x);

void __cdecl WindowsCCrashHandler::InvalidParameterHandler(const wchar_t *expression, const wchar_t *function,
                                                           const wchar_t *file, unsigned int line, uintptr_t pReserved)
{
    // Invalid parameter exception

    UNUSED(expression)
    UNUSED(function)
    UNUSED(file)
    UNUSED(line)
    UNUSED(pReserved)

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    TerminateProcess(GetCurrentProcess(), 1);
}

#    undef UNUSED

int __cdecl WindowsCCrashHandler::NewHandler(size_t)
{
    // 'new' operator memory allocation exception

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    TerminateProcess(GetCurrentProcess(), 1);
    return 0;
}

void WindowsCCrashHandler::SIGABRTHandler(int)
{
    // Caught SIGABRT C++ signal

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    TerminateProcess(GetCurrentProcess(), 1);
}

void WindowsCCrashHandler::SIGFPEHandler(int, int)
{
    // Floating point exception (SIGFPE)

    EXCEPTION_POINTERS *exception = (PEXCEPTION_POINTERS)_pxcptinfoptrs;

    HandleCommonException(GetCurrentProcess(), exception, 5);

    TerminateProcess(GetCurrentProcess(), 1);
}

void WindowsCCrashHandler::SIGILLHandler(int)
{
    // Illegal instruction (SIGILL)

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    TerminateProcess(GetCurrentProcess(), 1);
}

void WindowsCCrashHandler::SIGINTHandler(int)
{
    // Interruption (SIGINT)

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    TerminateProcess(GetCurrentProcess(), 1);
}

void WindowsCCrashHandler::SIGSEGVHandler(int)
{
    // Invalid storage access (SIGSEGV)

    PEXCEPTION_POINTERS exception = (PEXCEPTION_POINTERS)_pxcptinfoptrs;

    HandleAccessViolation(GetCurrentProcess(), exception, 5);

    TerminateProcess(GetCurrentProcess(), 1);
}

void WindowsCCrashHandler::SIGTERMHandler(int)
{
    // Termination request (SIGTERM)

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    TerminateProcess(GetCurrentProcess(), 1);
}

void WindowsCCrashHandler::SetProcessExceptionHandlder()
{
    // Install top-level SEH handler
    SetUnhandledExceptionFilter(UnhandledExceptionHandler);

    // Catch pure virtual function calls.
    // Because there is one _purecall_handler for the whole process,
    // calling this function immediately impacts all threads. The last
    // caller on any thread sets the handler.
    // http://msdn.microsoft.com/en-us/library/t296ys27.aspx
    _set_purecall_handler(PureCallHandler);

    // Catch new operator memory allocation exceptions
    _set_new_handler(NewHandler);

    // Catch invalid parameter exceptions.
    _set_invalid_parameter_handler(InvalidParameterHandler);

    // Set up C++ signal handlers

    _set_abort_behavior(_CALL_REPORTFAULT, _CALL_REPORTFAULT);

    // Catch an abnormal program termination
    signal(SIGABRT, SIGABRTHandler);

    // Catch illegal instruction handler
    signal(SIGINT, SIGINTHandler);

    // Catch a termination request
    signal(SIGTERM, SIGTERMHandler);
}

void WindowsCCrashHandler::SetThreadExceptionHandlder()
{
    // Catch terminate() calls.
    // In a multithreaded environment, terminate functions are maintained
    // separately for each thread. Each new thread needs to install its own
    // terminate function. Thus, each thread is in charge of its own termination handling.
    // http://msdn.microsoft.com/en-us/library/t6fk7h29.aspx
    set_terminate(TerminateHandler);

    // Catch unexpected() calls.
    // In a multithreaded environment, unexpected functions are maintained
    // separately for each thread. Each new thread needs to install its own
    // unexpected function. Thus, each thread is in charge of its own unexpected handling.
    // http://msdn.microsoft.com/en-us/library/h46t5b69.aspx
    set_unexpected(UnexpectedHandler);

    // Catch a floating point error
    typedef void (*sigh)(int);
    signal(SIGFPE, (sigh)SIGFPEHandler);

    // Catch an illegal instruction
    signal(SIGILL, SIGILLHandler);

    // Catch illegal storage access errors
    signal(SIGSEGV, SIGSEGVHandler);
}

#endif

void WindowsCCrashHandler::setup()
{
#if defined(_WIN32)
    SetProcessExceptionHandlder();
    SetThreadExceptionHandlder();
#endif
}

} // namespace free_kick::common