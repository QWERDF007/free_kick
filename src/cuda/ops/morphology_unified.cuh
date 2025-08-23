#pragma once

#include "opsdef.h"

#include "morph_common.cuh"

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <vector>

namespace free_kick::cuda::ops::unified {

// ==================== 辅助函数 ====================
// 计算共享内存大小
inline size_t calcSharedMemSize(dim3 block_dim, int se_w, int se_h, int anchor_x, int anchor_y)
{
    const int left   = anchor_x;
    const int right  = se_w - 1 - anchor_x;
    const int top    = anchor_y;
    const int bottom = se_h - 1 - anchor_y;
    return size_t(block_dim.x + left + right) * size_t(block_dim.y + top + bottom) * sizeof(uint8_t);
}

// 构建偏移列表（用于v2）
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

// ==================== 策略标签（用于模板特化） ====================
// v0: 直接全局内存访问
struct DirectAccessStrategy
{
};

// v1: 共享内存优化
struct SharedMemoryStrategy
{
};

// v2: 偏移列表优化
struct OffsetOptimizedStrategy
{
};

template<typename Strategy, typename SEType>
void morphologyEx(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h,
                  int img_stride, int op, const SEType *d_se, int n_offsets, int se_w, int se_h, int anchor_x,
                  int anchor_y, cudaStream_t stream);

template CUDA_OPS_API void morphologyEx<DirectAccessStrategy, uint8_t>(
    const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h, int img_stride, int op,
    const uint8_t *d_se, int n_offsets, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream);

template CUDA_OPS_API void morphologyEx<SharedMemoryStrategy, uint8_t>(
    const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h, int img_stride, int op,
    const uint8_t *d_se, int n_offsets, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream);

template CUDA_OPS_API void morphologyEx<OffsetOptimizedStrategy, int2>(
    const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h, int img_stride, int op,
    const int2 *d_se, int n_offsets, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream);

} // namespace free_kick::cuda::ops::unified