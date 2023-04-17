

#pragma once
#include "NvInferRuntime.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

// Use the CUDA runtime API to check for errors during kernel launches.
#define CUDA_CHECK(call)                                                                     \
    do                                                                                       \
    {                                                                                        \
        cudaError_t status = call;                                                           \
        if (status != cudaSuccess)                                                           \
        {                                                                                    \
            std::cerr << "CUDA error in file '" << __FILE__ << "' line " << __LINE__ << ": " \
                      << cudaGetErrorString(status) << std::endl;                            \
            std::exit(EXIT_FAILURE);                                                         \
        }                                                                                    \
    }                                                                                        \
    while (0)

// Execute inference

inline int64_t volume(const nvinfer1::Dims &d)
{
    return std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

// Compare two buffers for equality within a specified tolerance
inline bool buffersEqual(const void *buf1, const void *buf2, size_t size, float tolerance)
{
    const float *f1 = static_cast<const float *>(buf1);
    const float *f2 = static_cast<const float *>(buf2);

    std::vector<std::pair<float, float>> unmatched_buf;
    for (size_t i = 0; i < size; ++i)
    {
        if (std::abs(f1[i] - f2[i]) > tolerance)
        {
            unmatched_buf.emplace_back(f1[i], f2[i]);
            std::cout << f1[i] << " " << f2[i] << std::endl;
        }
    }

    return unmatched_buf.size() > 0 ? false : true;
}

template<typename T>
inline bool buffersEqual(const std::vector<T> &buf1, const std::vector<T> &buf2, size_t size, float tolerance)
{
    std::vector<std::pair<float, float>> unmatched_buf;
    for (size_t i = 0; i < size; ++i)
    {
        if (std::abs(buf1[i] - buf2[i]) > tolerance)
        {
            unmatched_buf.emplace_back(buf1[i], buf2[i]);
            std::cout << buf1[i] << " " << buf2[i] << std::endl;
        }
    }
    return unmatched_buf.size() > 0 ? false : true;
}

struct TRTObjDeleter
{
    template<typename T>
    void operator()(T *obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template<typename T>
using UniquePtr = std::unique_ptr<T, TRTObjDeleter>;

template<typename T>
UniquePtr<T> makeUnique(T *t)
{
    return UniquePtr<T>{t};
}

// 在GPU上分配内存
template<typename Dtype>
struct CudaBuffer
{
    size_t mSize; // 数据大小
    void  *mPtr;  // 指向GPU内存的指针

    /**
     * @brief 构造函数，分配GPU内存
     *
     * @param size 数据大小
     */
    CudaBuffer(size_t size)
    {
        mSize = size;
        CUDA_CHECK(cudaMalloc(&mPtr, sizeof(Dtype) * mSize));
    }

    /**
     * @brief 析构函数，释放GPU内存
     *
     */
    ~CudaBuffer()
    {
        if (mPtr != nullptr)
        {
            CUDA_CHECK(cudaFree(mPtr));
            mPtr = nullptr;
        }
    }
};

template<typename Dtype>
struct CudaHostBuffer
{
    size_t mSize; // 数据大小
    void  *mPtr;  // 指向GPU内存的指针

    /**
     * @brief 构造函数，分配GPU内存
     *
     * @param size 数据大小
     */
    CudaHostBuffer(size_t size)
    {
        mSize = size;
        CUDA_CHECK(cudaMallocHost(&mPtr, sizeof(Dtype) * mSize));
    }

    /**
     * @brief 析构函数，释放GPU内存
     *
     */
    ~CudaHostBuffer()
    {
        if (mPtr != nullptr)
        {
            CUDA_CHECK(cudaFreeHost(mPtr));
            mPtr = nullptr;
        }
    }
};
