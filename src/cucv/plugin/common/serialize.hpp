/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;

template<typename T>
inline void serialize_value(void **buffer, const T &value);

template<typename T>
inline void deserialize_value(const void **buffer, size_t *buffer_size, T *value);

namespace {

template<typename T, class Enable = void>
struct Serializer
{
};

template<typename T>
/**
 * @brief Serializer结构体的特化版本，用于序列化和反序列化算术类型、枚举类型和POD类型
 * 
 */
struct Serializer<
    T, typename std::enable_if<std::is_arithmetic<T>::value || std::is_enum<T>::value || std::is_pod<T>::value>::type>
{
    /**
     * @brief 计算序列化后的T类型数据的大小
     * 
     * @param value 要序列化的T类型数据
     * @return size_t 序列化后的数据大小
     */
    static size_t serialized_size(const T &)
    {
        return sizeof(T);
    }

    /**
     * @brief 序列化T类型数据
     * 
     * @param buffer 序列化后的数据存储位置
     * @param value 要序列化的T类型数据
     */
    static void serialize(void **buffer, const T &value)
    {
        ::memcpy(*buffer, &value, sizeof(T));
        reinterpret_cast<char *&>(*buffer) += sizeof(T);
    }

    /**
     * @brief 反序列化T类型数据
     * 
     * @param buffer 反序列化前的数据存储位置
     * @param buffer_size 反序列化前的数据大小
     * @param value 反序列化后的T类型数据
     */
    static void deserialize(const void **buffer, size_t *buffer_size, T *value)
    {
        assert(*buffer_size >= sizeof(T));
        ::memcpy(value, *buffer, sizeof(T));
        reinterpret_cast<const char *&>(*buffer) += sizeof(T);
        *buffer_size -= sizeof(T);
    }
};

template<>
struct Serializer<const char *>
{
    /**
     * @brief 计算序列化后的const char*类型数据的大小
     * 
     * @param value 要序列化的const char*类型数据
     * @return size_t 序列化后的数据大小
     */
    static size_t serialized_size(const char *value)
    {
        return strlen(value) + 1;
    }

    /**
     * @brief 序列化const char*类型数据
     * 
     * @param buffer 序列化后的数据存储位置
     * @param value 要序列化的const char*类型数据
     */
    static void serialize(void **buffer, const char *value)
    {
        ::strcpy(static_cast<char *>(*buffer), value);
        reinterpret_cast<char *&>(*buffer) += strlen(value) + 1;
    }

    /**
     * @brief 反序列化const char*类型数据
     * 
     * @param buffer 反序列化前的数据存储位置
     * @param buffer_size 反序列化前的数据大小
     * @param value 反序列化后的const char*类型数据
     */
    static void deserialize(const void **buffer, size_t *buffer_size, const char **value)
    {
        *value           = static_cast<const char *>(*buffer);
        size_t data_size = strnlen(*value, *buffer_size) + 1;
        assert(*buffer_size >= data_size);
        reinterpret_cast<const char *&>(*buffer) += data_size;
        *buffer_size -= data_size;
    }
};

template<typename T>
struct Serializer<std::vector<T>, typename std::enable_if<std::is_arithmetic<T>::value || std::is_enum<T>::value
                                                          || std::is_pod<T>::value>::type>
{
    static size_t serialized_size(const std::vector<T> &value)
    {
        return sizeof(value.size()) + value.size() * sizeof(T);
    }

    static void serialize(void **buffer, const std::vector<T> &value)
    {
        serialize_value(buffer, value.size());
        size_t nbyte = value.size() * sizeof(T);
        ::memcpy(*buffer, value.data(), nbyte);
        reinterpret_cast<char *&>(*buffer) += nbyte;
    }

    static void deserialize(const void **buffer, size_t *buffer_size, std::vector<T> *value)
    {
        size_t size;
        deserialize_value(buffer, buffer_size, &size);
        value->resize(size);
        size_t nbyte = value->size() * sizeof(T);
        assert(*buffer_size >= nbyte);
        ::memcpy(value->data(), *buffer, nbyte);
        reinterpret_cast<const char *&>(*buffer) += nbyte;
        *buffer_size -= nbyte;
    }
};

} // namespace

template<typename T>
inline size_t serialized_size(const T &value)
{
    return Serializer<T>::serialized_size(value);
}

template<typename T>
inline void serialize_value(void **buffer, const T &value)
{
    return Serializer<T>::serialize(buffer, value);
}

template<typename T>
inline void deserialize_value(const void **buffer, size_t *buffer_size, T *value)
{
    return Serializer<T>::deserialize(buffer, buffer_size, value);
}
