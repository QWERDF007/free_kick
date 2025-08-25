#pragma once

#include "opsdef.h"

#include "morph_common.cuh"
#include "morphology_cuda_v0.cuh"
#include "morphology_cuda_v1.cuh"
#include "morphology_cuda_v2.cuh"
#include "morphology_cuda_v3.cuh"
#include "morphology_cuda_v4.cuh"
#include "morphology_cuda_v5.cuh"

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <vector>

namespace free_kick::cuda::ops {

// ==================== 模板声明 ====================
template<typename Executor, typename SEType>
void morphologyEx(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, const int img_w,
                  const int img_h, const int img_stride, const int op, const SEType *d_se, const int n_offsets,
                  const int se_w, const int se_h, const int anchor_x, const int anchor_y, cudaStream_t stream);

// ==================== 模板显式实例化声明+导出 ====================
template CUDA_OPS_API void morphologyEx<v0<uint8_t>, uint8_t>(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp,
                                                              uint8_t *d_tmp2, const int img_w, const int img_h,
                                                              const int img_stride, const int op, const uint8_t *d_se,
                                                              const int n_offsets, const int se_w, const int se_h,
                                                              const int anchor_x, const int anchor_y,
                                                              cudaStream_t stream);

template CUDA_OPS_API void morphologyEx<v1<uint8_t>, uint8_t>(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp,
                                                              uint8_t *d_tmp2, const int img_w, const int img_h,
                                                              const int img_stride, const int op, const uint8_t *d_se,
                                                              const int n_offsets, const int se_w, const int se_h,
                                                              const int anchor_x, const int anchor_y,
                                                              cudaStream_t stream);

template CUDA_OPS_API void morphologyEx<v2<uint8_t>, uint8_t>(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp,
                                                              uint8_t *d_tmp2, const int img_w, const int img_h,
                                                              const int img_stride, const int op, const uint8_t *d_se,
                                                              const int n_offsets, const int se_w, const int se_h,
                                                              const int anchor_x, const int anchor_y,
                                                              cudaStream_t stream);

template CUDA_OPS_API void morphologyEx<v3<uint8_t>, uint8_t>(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp,
                                                              uint8_t *d_tmp2, const int img_w, const int img_h,
                                                              const int img_stride, const int op, const uint8_t *d_se,
                                                              const int n_offsets, const int se_w, const int se_h,
                                                              const int anchor_x, const int anchor_y,
                                                              cudaStream_t stream);

template CUDA_OPS_API void morphologyEx<v4<uint8_t>, int2>(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp,
                                                           uint8_t *d_tmp2, const int img_w, const int img_h,
                                                           const int img_stride, const int op, const int2 *d_se,
                                                           const int n_offsets, const int se_w, const int se_h,
                                                           const int anchor_x, const int anchor_y, cudaStream_t stream);

// ==================== v5模板特化声明+导出 ====================
template<>
CUDA_OPS_API void morphologyEx<v5<uint8_t>, int2>(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2,
                                                  const int img_w, const int img_h, const int img_stride, const int op,
                                                  const int2 *d_se, const int n_offsets, const int se_w, const int se_h,
                                                  const int anchor_x, const int anchor_y, cudaStream_t stream);

} // namespace free_kick::cuda::ops