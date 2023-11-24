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
#ifndef TRT_PLUGIN_H
#define TRT_PLUGIN_H
#include "NvInferPlugin.h"
#include "checkMacrosPlugin.h"

#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>

#define TRT_PLUGIN_API __declspec(dllexport)

typedef enum
{
    STATUS_SUCCESS         = 0,
    STATUS_FAILURE         = 1,
    STATUS_BAD_PARAM       = 2,
    STATUS_NOT_SUPPORTED   = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

namespace nvinfer1 {
namespace pluginInternal {
class BasePlugin : public IPluginV2
{
protected:
    // 设置插件命名空间
    void setPluginNamespace(const char *libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    // 获取插件命名空间
    const char *getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    // 插件命名空间
    std::string mNamespace;
};

class BaseCreator : public IPluginCreator
{
public:
    // 设置插件命名空间
    void setPluginNamespace(const char *libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    // 获取插件命名空间
    const char *getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    // 插件命名空间
    std::string mNamespace;
};
} // namespace pluginInternal

namespace plugin {
// 将值写入缓冲区
template<typename Type, typename BufferType>
void write(BufferType *&buffer, const Type &val)
{
    static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
    std::memcpy(buffer, &val, sizeof(Type));
    buffer += sizeof(Type);
}

// 从缓冲区读取值
template<typename OutType, typename BufferType>
OutType read(const BufferType *&buffer)
{
    static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
    OutType val{};
    std::memcpy(&val, static_cast<const void *>(buffer), sizeof(OutType));
    buffer += sizeof(OutType);
    return val;
}

// 获取TensorRT SM版本
inline int32_t getTrtSMVersionDec(int32_t smVersion)
{
    // 将SM89临时视为SM86。
    return (smVersion == 89) ? 86 : smVersion;
}

inline int32_t getTrtSMVersionDec(int32_t majorVersion, int32_t minorVersion)
{
    return getTrtSMVersionDec(majorVersion * 10 + minorVersion);
}

// 检查PluginFieldCollection中是否存在所有必需的字段名。
// 如果不存在，则抛出一个PluginError，其中包含缺少哪些字段的消息。
void validateRequiredAttributesExist(std::set<std::string> requiredFieldNames, const PluginFieldCollection *fc);

// 在GPU上分配内存
template<typename Dtype>
struct CudaBind
{
    size_t mSize; // 数据大小
    void  *mPtr;  // 指向GPU内存的指针

    /**
     * @brief 构造函数，分配GPU内存
     *
     * @param size 数据大小
     */
    CudaBind(size_t size)
    {
        mSize = size;
        PLUGIN_CUASSERT(cudaMalloc(&mPtr, sizeof(Dtype) * mSize));
    }

    /**
     * @brief 析构函数，释放GPU内存
     *
     */
    ~CudaBind()
    {
        if (mPtr != nullptr)
        {
            PLUGIN_CUASSERT(cudaFree(mPtr));
            mPtr = nullptr;
        }
    }
};

template<typename Dtype>
struct CudaHostBind
{
    size_t mSize; // 数据大小
    void  *mPtr;  // 指向GPU内存的指针

    /**
     * @brief 构造函数，分配GPU内存
     *
     * @param size 数据大小
     */
    CudaHostBind(size_t size)
    {
        mSize = size;
        PLUGIN_CUASSERT(cudaMallocHost(&mPtr, sizeof(Dtype) * mSize));
    }

    /**
     * @brief 析构函数，释放GPU内存
     *
     */
    ~CudaHostBind()
    {
        if (mPtr != nullptr)
        {
            PLUGIN_CUASSERT(cudaFreeHost(mPtr));
            mPtr = nullptr;
        }
    }
};

} // namespace plugin
} // namespace nvinfer1

#ifndef DEBUG

#    define PLUGIN_CHECK(status) \
        do                       \
        {                        \
            if (status != 0)     \
                abort();         \
        }                        \
        while (0)

#    define ASSERT_PARAM(exp)            \
        do                               \
        {                                \
            if (!(exp))                  \
                return STATUS_BAD_PARAM; \
        }                                \
        while (0)

#    define ASSERT_FAILURE(exp)        \
        do                             \
        {                              \
            if (!(exp))                \
                return STATUS_FAILURE; \
        }                              \
        while (0)

#    define CSC(call, err)                 \
        do                                 \
        {                                  \
            cudaError_t cudaStatus = call; \
            if (cudaStatus != cudaSuccess) \
            {                              \
                return err;                \
            }                              \
        }                                  \
        while (0)

#    define DEBUG_PRINTF(...) \
        do                    \
        {                     \
        }                     \
        while (0)

#else

#    define ASSERT_PARAM(exp)                                                         \
        do                                                                            \
        {                                                                             \
            if (!(exp))                                                               \
            {                                                                         \
                fprintf(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__); \
                return STATUS_BAD_PARAM;                                              \
            }                                                                         \
        }                                                                             \
        while (0)

#    define ASSERT_FAILURE(exp)                                                     \
        do                                                                          \
        {                                                                           \
            if (!(exp))                                                             \
            {                                                                       \
                fprintf(stderr, "Failure - " #exp ", %s:%d\n", __FILE__, __LINE__); \
                return STATUS_FAILURE;                                              \
            }                                                                       \
        }                                                                           \
        while (0)

#    define CSC(call, err)                                                                          \
        do                                                                                          \
        {                                                                                           \
            cudaError_t cudaStatus = call;                                                          \
            if (cudaStatus != cudaSuccess)                                                          \
            {                                                                                       \
                printf("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus)); \
                return err;                                                                         \
            }                                                                                       \
        }                                                                                           \
        while (0)

#    define PLUGIN_CHECK(status)                                                                      \
        {                                                                                             \
            if (status != 0)                                                                          \
            {                                                                                         \
                DEBUG_PRINTF("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
                abort();                                                                              \
            }                                                                                         \
        }

#    define DEBUG_PRINTF(...)    \
        do                       \
        {                        \
            printf(__VA_ARGS__); \
        }                        \
        while (0)

#endif // DEBUG

#endif // TRT_PLUGIN_H
