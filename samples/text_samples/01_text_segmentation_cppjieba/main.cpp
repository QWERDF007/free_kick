#include <Windows.h>
#include <cppjieba/Jieba.hpp>
#include <psapi.h>

#include <chrono>
#include <future>

std::vector<std::string> dicts{
    "目标检测", "目标检测-yolov8",  "目标检测-yolov5",       "实例分割-yolov8", "语义分割",
    "目标",     "object detection", "semantic segmentation",
};

std::shared_ptr<cppjieba::Jieba> initJieba()
{
    return std::make_shared<cppjieba::Jieba>();
}

void getInfo()
{
    HANDLE hProcess = GetCurrentProcess();
    if (hProcess == NULL)
    {
        std::cerr << "Error opening process." << std::endl;
        return;
    }

    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(hProcess, (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc)))
    {
        std::cout << "Working Set Size: " << pmc.WorkingSetSize / 1024.0 / 1024.0 << " MB" << std::endl;
        std::cout << "Peak Working Set Size: " << pmc.PeakWorkingSetSize / 1024.0 / 1024.0 << " MB" << std::endl;
        std::cout << "Private Usage: " << pmc.PrivateUsage / 1024.0 / 1024.0 << " MB" << std::endl;
    }
    else
    {
        std::cerr << "Error getting process memory info." << std::endl;
    }

    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo))
    {
        std::cout << "Memory Load: " << memInfo.dwMemoryLoad << "%" << std::endl;
        std::cout << "Total Physical: " << memInfo.ullTotalPhys / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
        std::cout << "Available Physical: " << memInfo.ullAvailPhys / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
        std::cout << "Used Physical: " << (memInfo.ullTotalPhys - memInfo.ullAvailPhys) / 1024.0 / 1024.0 / 1024.0
                  << " GB" << std::endl;
    }

    return;
}

int main(int argc, char **argv)
{
    std::cout << "memory usage: " << std::endl;
    getInfo();
    std::cout << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    auto res = std::async(std::launch::async, initJieba);

    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);
    std::cout << "async init elapsed: " << duration.count() << " ms" << std::endl;

    auto status = res.wait_for(std::chrono::seconds(1));

    std::cout << "ready: " << bool(status == std::future_status::ready) << std::endl;

    std::cout << "res valid: " << res.valid() << std::endl;

    start = std::chrono::high_resolution_clock::now();

    auto jieba = res.get();
    std::cout << "res valid: " << res.valid() << std::endl;

    end      = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end - start);

    std::cout << "get elapsed: " << duration.count() << " ms" << std::endl;

    std::cout << std::endl;
    std::cout << "memory usage: " << std::endl;
    start = std::chrono::high_resolution_clock::now();
    getInfo();
    end      = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end - start);
    std::cout << std::endl;

    std::cout << "memory query elapsed: " << duration.count() << " ms" << std::endl;

    for (const std::string &s : dicts)
    {
        start = std::chrono::high_resolution_clock::now();

        std::vector<std::string> words;
        // jieba->Cut(s, words, true);

        jieba->extractor.Extract(s, words, 5);

        end      = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double, std::milli>(end - start);

        std::cout << s << " cut size: " << words.size() << " cut elapsed: " << duration.count() << " ms" << std::endl;
        for (auto &word : words)
        {
            std::cout << word << " ";
        }
        std::cout << std::endl;
    }

    // Sleep(10000);
    return 0;
}