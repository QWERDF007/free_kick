#include "WindowsCrashHandler.h"

// clang-format off
#if defined(_WIN32)

#include <rtcapi.h>
#include <DbgHelp.h>
#include <psapi.h>
#include <signal.h>
#include <new.h>
#include <tchar.h>

#include <iostream>

#ifndef _AddressOfReturnAddress

// Taken from: http://msdn.microsoft.com/en-us/library/s975zw7k(VS.71).aspx
#ifdef __cplusplus
    #define EXTERNC extern "C"
#else
    #define EXTERNC
#endif

// _ReturnAddress and _AddressOfReturnAddress should be prototyped before use 
EXTERNC void * _AddressOfReturnAddress(void);
EXTERNC void * _ReturnAddress(void);

#endif 

namespace free_kick::common {

// https://win32easy.blogspot.com/2011/03/exception-handling-inform-your-users_26.html

#define EX_CASE(code)     \
    case code:            \
        return #code;

/**
 * @brief 获取异常名称, 通过宏将异常代码转换为字符串
 * @param code 
 * @return LPCSTR 
 */
LPCSTR CCrashHandler::GetExceptionName(DWORD code)
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

void CCrashHandler::GetExceptionPointers(DWORD dwExceptionCode, EXCEPTION_POINTERS **ppExceptionPointers)
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

HMODULE CCrashHandler::GetExceptionModule(HANDLE process, LPVOID address, LPSTR module_name)
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
            else
            {
                if ((DWORD)module_list[cur_module] < (DWORD)module_list[i])
                    cur_module = i;
            }
        }
    }

    if (cur_module == -1)
        return NULL;

    if (!GetModuleFileName(module_list[cur_module], module_name, MAX_PATH))
        return NULL;

    return module_list[cur_module];
}

void CCrashHandler::PrintStackTrace(HANDLE process, const ULONG frames_to_skip)
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
    // 设置符号选项，确保我们获取源文件信息
    sym_options |= SYMOPT_LOAD_LINES;
    sym_options |= SYMOPT_DEBUG;
    SymSetOptions(sym_options);

    std::cerr << "\nTraceback (most recent call last): " << std::endl;
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
        {
            std::cerr << "  File \"" << line.FileName << "\", line " << line.LineNumber << " in " << symbol->Name
                      << std::endl;
            std::cerr << "    " << symbol->Name << std::endl;
        }
        // 清理符号缓存
        free(symbol);
    }
    SymCleanup(process);
}

void CCrashHandler::HandleAccessViolation(HANDLE process, LPEXCEPTION_POINTERS exception, const ULONG frames_to_skip)
{
    char  message[MAX_PATH + 512];
    char  module[MAX_PATH];
    char *module_name = NULL;
    if (GetExceptionModule(process, exception->ExceptionRecord->ExceptionAddress, module))
        module_name = module;
    else
        module_name = "Unknown module!";

    DWORD code_base = (DWORD)GetModuleHandle(NULL);
    DWORD offset    = (DWORD)exception->ExceptionRecord->ExceptionAddress - code_base;

    char *access_type = NULL;
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

    const char *exception_name = GetExceptionName(exception->ExceptionRecord->ExceptionCode);
    sprintf_s(message,
              "An exception has occured which was not handled!\nCode: %s\nModule: %s\n"
              "The thread %u tried to %s memory at address 0x%08X which is inaccessible!\n"
              "Offset: 0x%08X\nCodebase: 0x%08X",
              exception_name, module_name, GetCurrentThreadId(), access_type,
              exception->ExceptionRecord->ExceptionInformation[1], offset, code_base);
    std::cerr << message << std::endl;

#    ifdef _DEBUG
    PrintStackTrace(process, frames_to_skip);
#    endif
}

void CCrashHandler::HandleCommonException(HANDLE process, LPEXCEPTION_POINTERS exception, const ULONG frames_to_skip)
{
    char  message[MAX_PATH + 255];
    char  module[MAX_PATH];
    char *module_name = NULL;
    if (GetExceptionModule(process, exception->ExceptionRecord->ExceptionAddress, module))
        module_name = module;
    else
        module_name = "Unknown module!";
    const char *exception_name = GetExceptionName(exception->ExceptionRecord->ExceptionCode);
    sprintf_s(message, "An exception has occured which was not handled!\nCode: %s\nModule: %s", exception_name,
              module_name);
    std::cerr << message << std::endl;

#    ifdef _DEBUG
    PrintStackTrace(process, frames_to_skip);
#    endif
}

void CCrashHandler::CreateMiniDump(EXCEPTION_POINTERS *pExcPtrs)
{
    HMODULE                        hDbgHelp = NULL;
    HANDLE                         hFile    = NULL;
    MINIDUMP_EXCEPTION_INFORMATION mei;
    MINIDUMP_CALLBACK_INFORMATION  mci;

    // Load dbghelp.dll
    hDbgHelp = LoadLibrary(_T("dbghelp.dll"));
    if (hDbgHelp == NULL)
    {
        // Error - couldn't load dbghelp.dll
        return;
    }

    // Create the minidump file
    hFile = CreateFile(_T("crashdump.dmp"), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (hFile == INVALID_HANDLE_VALUE)
    {
        // Couldn't create file
        return;
    }

    // Write minidump to the file
    mei.ThreadId          = GetCurrentThreadId();
    mei.ExceptionPointers = pExcPtrs;
    mei.ClientPointers    = FALSE;
    mci.CallbackRoutine   = NULL;
    mci.CallbackParam     = NULL;

    typedef BOOL(WINAPI * LPMINIDUMPWRITEDUMP)(HANDLE hProcess, DWORD ProcessId, HANDLE hFile, MINIDUMP_TYPE DumpType,
                                               CONST PMINIDUMP_EXCEPTION_INFORMATION   ExceptionParam,
                                               CONST PMINIDUMP_USER_STREAM_INFORMATION UserEncoderParam,
                                               CONST PMINIDUMP_CALLBACK_INFORMATION    CallbackParam);

    LPMINIDUMPWRITEDUMP pfnMiniDumpWriteDump = (LPMINIDUMPWRITEDUMP)GetProcAddress(hDbgHelp, "MiniDumpWriteDump");
    if (!pfnMiniDumpWriteDump)
    {
        // Bad MiniDumpWriteDump function
        return;
    }

    HANDLE hProcess    = GetCurrentProcess();
    DWORD  dwProcessId = GetCurrentProcessId();

    BOOL bWriteDump = pfnMiniDumpWriteDump(hProcess, dwProcessId, hFile, MiniDumpNormal, &mei, NULL, &mci);

    if (!bWriteDump)
    {
        // Error writing dump.
        return;
    }

    // Close file
    CloseHandle(hFile);

    // Unload dbghelp.dll
    FreeLibrary(hDbgHelp);
}

LONG WINAPI CCrashHandler::UnhandledExceptionHandler(LPEXCEPTION_POINTERS exception)
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

    CreateMiniDump(exception);

    TerminateProcess(GetCurrentProcess(), 1);

    return EXCEPTION_EXECUTE_HANDLER;
}

void __cdecl CCrashHandler::TerminateHandler()
{
    // Abnormal program termination (terminate() function was called)

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    CreateMiniDump(exception);

    TerminateProcess(GetCurrentProcess(), 1);
}

void __cdecl CCrashHandler::UnexpectedHandler()
{
    // Unexpected error (unexpected() function was called)

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    CreateMiniDump(exception);

    TerminateProcess(GetCurrentProcess(), 1);
}

void CCrashHandler::PureCallHandler()
{
    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    CreateMiniDump(exception);

    TerminateProcess(GetCurrentProcess(), 1);
}

#    define UNUSED(x) (void)(x);

void __cdecl CCrashHandler::InvalidParameterHandler(const wchar_t *expression, const wchar_t *function,
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

    CreateMiniDump(exception);

    TerminateProcess(GetCurrentProcess(), 1);
}

#    undef UNUSED

int __cdecl CCrashHandler::NewHandler(size_t)
{
    // 'new' operator memory allocation exception

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    CreateMiniDump(exception);

    TerminateProcess(GetCurrentProcess(), 1);
    return 0;
}

void CCrashHandler::SIGABRTHandler(int)
{
    // Caught SIGABRT C++ signal

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    CreateMiniDump(exception);

    TerminateProcess(GetCurrentProcess(), 1);
}

void CCrashHandler::SIGFPEHandler(int, int)
{
    // Floating point exception (SIGFPE)

    EXCEPTION_POINTERS *exception = (PEXCEPTION_POINTERS)_pxcptinfoptrs;

    HandleCommonException(GetCurrentProcess(), exception, 5);

    CreateMiniDump(exception);

    TerminateProcess(GetCurrentProcess(), 1);
}

void CCrashHandler::SIGILLHandler(int)
{
    // Illegal instruction (SIGILL)

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    CreateMiniDump(exception);

    TerminateProcess(GetCurrentProcess(), 1);
}

void CCrashHandler::SIGINTHandler(int)
{
    // Interruption (SIGINT)

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    CreateMiniDump(exception);

    TerminateProcess(GetCurrentProcess(), 1);
}

void CCrashHandler::SIGSEGVHandler(int)
{
    // Invalid storage access (SIGSEGV)

    PEXCEPTION_POINTERS exception = (PEXCEPTION_POINTERS)_pxcptinfoptrs;

    HandleAccessViolation(GetCurrentProcess(), exception, 5);

    CreateMiniDump(exception);

    TerminateProcess(GetCurrentProcess(), 1);
}

void CCrashHandler::SIGTERMHandler(int)
{
    // Termination request (SIGTERM)

    EXCEPTION_POINTERS *exception = NULL;
    GetExceptionPointers(0, &exception);

    HandleCommonException(GetCurrentProcess(), exception, 5);

    CreateMiniDump(exception);

    TerminateProcess(GetCurrentProcess(), 1);
}

void CCrashHandler::SetProcessExceptionHandlder()
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

void CCrashHandler::SetThreadExceptionHandlder()
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

void CCrashHandler::setup()
{
#if defined(_WIN32)
    SetProcessExceptionHandlder();
    SetThreadExceptionHandlder();
#endif
}

} // namespace free_kick::common