#pragma once

#include "opsdef.h"

#include "morph_common.cuh"
#include "morphology_cuda_v0.cuh"
#include "morphology_cuda_v1.cuh"
#include "morphology_cuda_v2.cuh"

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <vector>

namespace free_kick::cuda::ops {

// ==================== 辅助函数 ====================

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

// ==================== 模板 ====================
template<typename Executor, typename SEType>
void morphologyEx(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, const int img_w,
                  const int img_h, const int img_stride, const int op, const SEType *d_se, const int n_offsets,
                  const int se_w, const int se_h, const int anchor_x, const int anchor_y, cudaStream_t stream);

// ==================== 模板特化+导出 ====================
template CUDA_OPS_API void morphologyEx<v0::DirectAccessExecutor<uint8_t>, uint8_t>(
    const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, const int img_w, const int img_h,
    const int img_stride, const int op, const uint8_t *d_se, const int n_offsets, const int se_w, const int se_h,
    const int anchor_x, const int anchor_y, cudaStream_t stream);

template CUDA_OPS_API void morphologyEx<v1::SharedMemoryExecutor<uint8_t>, uint8_t>(
    const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, const int img_w, const int img_h,
    const int img_stride, const int op, const uint8_t *d_se, const int n_offsets, const int se_w, const int se_h,
    const int anchor_x, const int anchor_y, cudaStream_t stream);

template CUDA_OPS_API void morphologyEx<v2::OffsetOptimizedExecutor<uint8_t>, int2>(
    const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, const int img_w, const int img_h,
    const int img_stride, const int op, const int2 *d_se, const int n_offsets, const int se_w, const int se_h,
    const int anchor_x, const int anchor_y, cudaStream_t stream);

} // namespace free_kick::cuda::ops