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
typename std::enable_if<std::is_floating_point<T>::value, std::vector<T>>::type random(T min, T max, size_t count)
{
    std::vector<T> result;
    result.reserve(count);

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(min, max);

    for (size_t i = 0; i < count; ++i)
    {
        result.push_back(dis(gen));
    }

    return result;
}

/**
 * @brief 生成一组整形随机数
 */
template<typename T>
typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type random(T min, T max, size_t count)
{
    std::vector<T> result;
    result.reserve(count);

    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(min, max);

    for (size_t i = 0; i < count; ++i)
    {
        result.push_back(dis(gen));
    }

    return result;
}

} // namespace free_kick::utils