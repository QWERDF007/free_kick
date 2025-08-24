#pragma once

#include "opsdef.h"

#include "morph_common.cuh"
#include "morphology_unified.cuh"

#include <opencv2/opencv.hpp>

#include <cstdint>

namespace free_kick::cuda::ops::v0 {

// v0: 直接全局内存访问
template<typename T>
struct DirectAccessExecutor
{
    // 结构元素 se 为大小 se_w x se_h 的二值掩码，anchor_x/anchor_y 为锚点（通常为中心）
    // 边界采用 replicate（坐标 clamp）
    template<typename Reducer>
    __device__ void operator()(const Reducer &reducer, const T *__restrict__ in, T *__restrict__ out, const int img_w,
                               const int img_h, const int img_stride, const uint8_t *__restrict__ d_se,
                               const int n_offsets, const int se_w, const int se_h, const int anchor_x,
                               const int anchor_y) const
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= img_w || y >= img_h)
            return;

        T acc = reducer.init();

        // 遍历结构元素
        for (int ky = 0; ky < se_h; ++ky)
        {
            int            iy     = clampIndex(y + ky - anchor_y, 0, img_h);
            const uint8_t *se_row = d_se + ky * se_w;

            for (int kx = 0; kx < se_w; ++kx)
            {
                if (se_row[kx])
                {
                    int ix  = clampIndex(x + kx - anchor_x, 0, img_w);
                    T   val = in[iy * img_stride + ix];
                    acc     = reducer.reduce(acc, val);
                }
            }
        }

        out[y * img_stride + x] = acc;
    }

    inline size_t calcSharedMemSize(dim3, int, int, int, int)
    {
        return 0;
    }
};

} // namespace free_kick::cuda::ops::v0