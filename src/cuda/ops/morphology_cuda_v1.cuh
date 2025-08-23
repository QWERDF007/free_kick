#pragma once

#include "opsdef.h"

#include "morph_common.cuh"

#include <opencv2/opencv.hpp>

#include <cstdint>

namespace free_kick::cuda::ops::v1 {

// v1: 共享内存优化
struct SharedMemoryStrategy
{
};

template<typename T, typename Reducer>
__global__ void morphKernel(const T *__restrict__ in, T *__restrict__ out, const int img_w, const int img_h,
                            const int img_stride, const uint8_t *__restrict__ d_se, const int n_offsets, const int se_w,
                            const int se_h, const int anchor_x, const int anchor_y)
{
    extern __shared__ unsigned char smem_u8[];

    T *smem = reinterpret_cast<T *>(smem_u8);

    const int left   = anchor_x;
    const int right  = se_w - 1 - anchor_x;
    const int top    = anchor_y;
    const int bottom = se_h - 1 - anchor_y;

    const int tile_w = blockDim.x + left + right;
    const int tile_h = blockDim.y + top + bottom;

    const int block_x = blockIdx.x * blockDim.x;
    const int block_y = blockIdx.y * blockDim.y;

    // 共享内存加载（分块遍历）
    for (int yy = threadIdx.y; yy < tile_h; yy += blockDim.y)
    {
        int      gy     = clampIndex(block_y + yy - top, 0, img_h);
        const T *in_row = in + gy * img_stride;

        for (int xx = threadIdx.x; xx < tile_w; xx += blockDim.x)
        {
            int gx                 = clampIndex(block_x + xx - left, 0, img_w);
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

    // 以共享内存为中心并依据掩码进行 reduce
    const int sx = threadIdx.x + left;
    const int sy = threadIdx.y + top;

    for (int ky = 0; ky < se_h; ++ky)
    {
        const int      row    = (sy + (ky - anchor_y)) * tile_w;
        const uint8_t *se_row = d_se + ky * se_w;
        for (int kx = 0; kx < se_w; ++kx)
        {
            if (se_row[kx])
            {
                acc = reducer.reduce(acc, smem[row + (sx + (kx - anchor_x))]);
            }
        }
    }

    out[y * img_stride + x] = acc;
}

} // namespace free_kick::cuda::ops::v1