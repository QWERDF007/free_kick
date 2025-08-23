#pragma once

#include <cuda_runtime.h>

#include <limits>

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

} // namespace free_kick::cuda::ops