#pragma once

#include <random>
#include <type_traits>
#include <vector>

namespace free_kick::utils {

template<typename T>
T clamp(T value, T min, T max)
{
    return std::max(min, std::min(max, value));
}

/**
 * @brief 生成一组浮点随机数
 * 
 * @tparam T 
 * @param min
 * @param max 
 * @param count 
 * @return std::vector<T> 
 */
template<typename T>
std::vector<T> random(T min, T max, size_t count)
{
    std::vector<T> result;
    result.reserve(count);

    std::random_device rd;
    std::mt19937       gen(rd());
    auto               dis = uniform_distribution<T>(min, max);

    for (size_t i = 0; i < count; ++i)
    {
        result.push_back(dis(gen));
    }

    return result;
}

template<typename T>
std::enable_if_t<std::is_integral<T>::value, std::uniform_int_distribution<>> uniform_distribution(T min, T max)
{
    return std::uniform_int_distribution<>(min, max);
}

template<typename T>
std::enable_if_t<std::is_floating_point<T>::value, std::uniform_real_distribution<>> uniform_distribution(T min, T max)
{
    return std::uniform_real_distribution<>(min, max);
}

} // namespace free_kick::utils