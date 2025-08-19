#pragma once

#include "opsdef.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <cstdint>

namespace free_kick::cuda::ops::v2 {

// Host 侧：构建有效偏移列表（只保留为 1 的位置）
inline std::vector<int2> build_se_offsets(const uint8_t *h_se, int se_w, int se_h, int anchor_x, int anchor_y)
{
    std::vector<int2> offs;
    offs.reserve(se_w * se_h);
    for (int ky = 0; ky < se_h; ++ky)
    {
        for (int kx = 0; kx < se_w; ++kx)
        {
            if (h_se[ky * se_w + kx])
            {
                offs.push_back(make_int2(kx - anchor_x, ky - anchor_y));
            }
        }
    }
    return offs;
}

// -------------------- 统一的形态学操作接口 --------------------
CUDA_OPS_API void morphologyEx(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w,
                               int img_h, int img_stride, const int op, const int2 *d_se, int n_offsets, int se_w,
                               int se_h, int anchor_x, int anchor_y, dim3 block_dim, cudaStream_t stream);

} // namespace free_kick::cuda::ops::v2