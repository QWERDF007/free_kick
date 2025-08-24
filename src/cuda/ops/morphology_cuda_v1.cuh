#pragma once

#include "morph_common.cuh"

namespace free_kick::cuda::ops {

// v1: 无分支 clamp
template<typename T>
struct v1
{
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
            int            iy     = clampIndexNoBranch(y + ky - anchor_y, 0, img_h);
            const uint8_t *se_row = d_se + ky * se_w;

            for (int kx = 0; kx < se_w; ++kx)
            {
                if (se_row[kx])
                {
                    int ix  = clampIndexNoBranch(x + kx - anchor_x, 0, img_w);
                    T   val = in[iy * img_stride + ix];
                    acc     = reducer.reduce(acc, val);
                }
            }
        }

        out[y * img_stride + x] = acc;
    }

    size_t getSharedMemSize(dim3, int, int, int, int)
    {
        return 0;
    }
};

} // namespace free_kick::cuda::ops