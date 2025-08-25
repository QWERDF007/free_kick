#pragma once

#include "opsdef.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <vector>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16
#define THREAD_SIZE  256

namespace free_kick::cuda::ops {

template<typename T>
struct MaxReducer
{
    __device__ __forceinline__ T init() const
    {
        return std::numeric_limits<T>::min();
    }

    __device__ __forceinline__ T reduce(T a, T b) const
    {
        return a > b ? a : b;
    }
};

template<typename T>
struct MinReducer
{
    __device__ __forceinline__ T init() const
    {
        return std::numeric_limits<T>::max();
    }

    __device__ __forceinline__ T reduce(T a, T b) const
    {
        return a < b ? a : b;
    }
};

inline static int divUp(int a, int b)
{
    return (a + b - 1) / b;
}

// -------------------- 简单逐点差值核（带饱和） --------------------
template<typename T, T MIN_VAL, T MAX_VAL>
__global__ void sub_clamp_kernel(const T *a, const T *b, T *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int v = int(a[i]) - int(b[i]);
        v     = v < MIN_VAL ? MIN_VAL : (v > MAX_VAL ? MAX_VAL : v);
        c[i]  = static_cast<T>(v);
    }
}

__device__ __forceinline__ int clampIndex(int x, int low, int high_exclusive)
{
    if (x < low)
        return low;
    if (x >= high_exclusive)
        return high_exclusive - 1;
    return x;
}

// 无分支 clamp：使用 min/max 组合，编译为 IMNMX 指令，无 warp 分歧
__device__ __forceinline__ int clampIndexNoBranch(int x, int low, int high_exclusive)
{
    // x      = max(x, low); // 编译器会用 IMNMX；
    // int hi = high_exclusive - 1;
    // x      = min(x, hi);
    x      = (x > low ? x : low);
    int hi = high_exclusive - 1;
    x      = (x < hi ? x : hi);
    return x;
}

// 计算共享内存大小
template<typename T>
inline size_t calcSharedMemSize(dim3 block_dim, int se_w, int se_h, int anchor_x, int anchor_y)
{
    const int left   = anchor_x;
    const int right  = se_w - 1 - anchor_x;
    const int top    = anchor_y;
    const int bottom = se_h - 1 - anchor_y;
    return size_t(block_dim.x + left + right) * size_t(block_dim.y + top + bottom) * sizeof(T);
}

// 构建偏移列表（用于v4）
inline std::vector<int2> buildOffsetList(const uint8_t *h_se, int se_w, int se_h, int anchor_x, int anchor_y)
{
    std::vector<int2> offsets;
    offsets.reserve(se_w * se_h);
    for (int ky = 0; ky < se_h; ++ky)
    {
        for (int kx = 0; kx < se_w; ++kx)
        {
            if (h_se[ky * se_w + kx])
            {
                offsets.push_back(make_int2(kx - anchor_x, ky - anchor_y));
            }
        }
    }
    return offsets;
}

} // namespace free_kick::cuda::ops