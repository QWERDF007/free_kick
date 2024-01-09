#pragma once
#include <vector>

namespace free_kick::utils {
/**
 * @brief 寻找出现次数最多的元素
 * 
 * @param arr 待寻找的数组
 * @param element 出现次数最多的元素
 * @param freq 出现次数
 */
void findMostFrequentElement(const std::vector<int> &arr, int &element, int &freq);
} // namespace free_kick::utils