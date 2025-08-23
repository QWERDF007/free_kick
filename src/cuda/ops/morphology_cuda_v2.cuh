#pragma once

#include "opsdef.h"

#include "morph_common.cuh"

#include <opencv2/opencv.hpp>

#include <cstdint>

namespace free_kick::cuda::ops::v2 {

// v2: 偏移列表优化
struct OffsetOptimizedStrategy
{
};

template<typename T, typename Reducer>
__global__ void morphKernel(const T *__restrict__ in, T *__restrict__ out, const int img_w, const int img_h,
                            const int img_stride, const int2 *__restrict__ d_se, const int n_offsets, const int se_w,
                            const int se_h, const int anchor_x, const int anchor_y)
{
    extern __shared__ unsigned char smem_u8[];
    T                              *smem = reinterpret_cast<T *>(smem_u8);

    const int left   = anchor_x;
    const int right  = se_w - 1 - anchor_x;
    const int top    = anchor_y;
    const int bottom = se_h - 1 - anchor_y;

    const int tile_w = blockDim.x + left + right;
    const int tile_h = blockDim.y + top + bottom;

    const int block_x = blockIdx.x * blockDim.x;
    const int block_y = blockIdx.y * blockDim.y;

    // 将 tile 区域加载到共享内存（无分支 clamp）
    for (int yy = threadIdx.y; yy < tile_h; yy += blockDim.y)
    {
        int gy = clampIndexNoBranch(block_y + yy - top, 0, img_h);

        const T *in_row = in + gy * img_stride;

        for (int xx = threadIdx.x; xx < tile_w; xx += blockDim.x)
        {
            int gx                 = clampIndexNoBranch(block_x + xx - left, 0, img_w);
            smem[yy * tile_w + xx] = in_row[gx];
        }
    }
    __syncthreads();

    const int x = block_x + threadIdx.x;
    const int y = block_y + threadIdx.y;
    if (x >= img_w || y >= img_h)
        return;

    Reducer reducer;
    T       acc = reducer.init();

    // 当前像素在共享内存中的中心位置
    const int sx = threadIdx.x + left;
    const int sy = threadIdx.y + top;

    // 遍历有效偏移（已预处理，无需 if 掩码判断）
    // 共享内存访问范围始终在 [0, tile_w/tile_h) 内，无需再次 clamp
    const int row_stride = tile_w;

    // 使用预计算的偏移列表
    for (int i = 0; i < n_offsets; ++i)
    {
        const int ox  = d_se[i].x;
        const int oy  = d_se[i].y;
        const int idx = (sy + oy) * row_stride + (sx + ox);
        acc           = reducer.reduce(acc, smem[idx]);
    }

    out[y * img_stride + x] = acc;
}
} // namespace free_kick::cuda::ops::v2