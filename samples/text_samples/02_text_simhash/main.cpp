
#include "simhash/Simhasher.hpp"

#include <chrono>
#include <future>
#include <iostream>

std::vector<std::string> dicts{
    "目标检测", "目标检测-yolov8",  "目标检测-yolov5",       "实例分割-yolov8", "语义分割",
    "目标",     "object detection", "semantic segmentation",
};

// std::shared_ptr<cppjieba::Jieba> initJieba()
// {
//     return std::make_shared<cppjieba::Jieba>();
// }

int main(int argc, char **argv)
{
    auto start = std::chrono::high_resolution_clock::now();

    // auto res = std::async(std::launch::async, initJieba);
    simhash::Simhasher simhasher;

    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);

    size_t topN = 5;

    std::vector<uint64_t> u64s;
    for (const std::string &dict : dicts)
    {
        uint64_t u64 = 0;

        std::vector<std::pair<std::string, double>> res;

        simhasher.extract(dict, res, topN);

        simhasher.make(dict, topN, u64);
        std::cout << "res: " << res << std::endl;
        std::string bstr;
        simhash::Simhasher::toBinaryString(u64, bstr);
        std::cout << "dict: " << dict << " u64: " << bstr << std::endl;
        u64s.push_back(u64);
    }

    for (size_t i = 0; i < u64s.size(); i++)
    {
        for (size_t j = 0; j < u64s.size(); j++)
        {
            size_t      n   = simhash::Simhasher::hammingDistance(u64s[i], u64s[j]);
            uint64_t    u64 = u64s[i];
            std::string bstr;
            simhash::Simhasher::toBinaryString(u64, bstr);
            std::cout << dicts[i] << " u64: " << u64 << " binary: " << bstr << std::endl;
            u64 = u64s[j];
            simhash::Simhasher::toBinaryString(u64, bstr);
            std::cout << dicts[j] << " u64: " << u64 << " binary: " << bstr << std::endl;
            std::cout << "hamming distance between " << dicts[i] << " and " << dicts[j] << " is " << n << std::endl;
            std::cout << std::endl;
        }
    }

    // Sleep(10000);
    return 0;
}