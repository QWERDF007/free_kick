#include "algo_utils.h"

#include <algorithm>
#include <unordered_map>

namespace free_kick::utils {
void findMostFrequentElement(const std::vector<int> &arr, int &element, int &freq)
{
    if (arr.empty())
    {
        element = std::numeric_limits<int>::min();
        freq    = 0;
        return;
    }
    std::unordered_map<int, int> freq_map;
    for (int i : arr)
    {
        freq_map[i]++;
    }
    auto res = std::max_element(freq_map.begin(), freq_map.end(),
                                [](const std::pair<int, int> &a, const std::pair<int, int> &b)
                                { return a.second < b.second; });
    element  = res->first;
    freq     = res->second;
}
} // namespace free_kick::utils