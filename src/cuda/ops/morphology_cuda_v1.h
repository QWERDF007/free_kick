#pragma once

#include "opsdef.h"

#include "morph_common.cuh"

#include <opencv2/opencv.hpp>

#include <cstdint>

namespace free_kick::cuda::ops::v1 {

// -------------------- Host 端接口声明 --------------------

// -------------------- uint8_t 专用实现声明（带掩码结构元素） --------------------
CUDA_OPS_API void morphDilate_u8_masked(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride,
                                        const uint8_t *d_se, int se_w, int se_h, int anchor_x, int anchor_y,
                                        dim3 block_dim, cudaStream_t stream);
CUDA_OPS_API void morphErode_u8_masked(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride,
                                       const uint8_t *d_se, int se_w, int se_h, int anchor_x, int anchor_y,
                                       dim3 block_dim, cudaStream_t stream);
CUDA_OPS_API void morphOpen_u8_masked(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_out, int img_w, int img_h,
                                      int img_stride, const uint8_t *d_se, int se_w, int se_h, int anchor_x,
                                      int anchor_y, dim3 block_dim, cudaStream_t stream);
CUDA_OPS_API void morphClose_u8_masked(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_out, int img_w, int img_h,
                                       int img_stride, const uint8_t *d_se, int se_w, int se_h, int anchor_x,
                                       int anchor_y, dim3 block_dim, cudaStream_t stream);
CUDA_OPS_API void morphTopHat_u8_masked(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_open, uint8_t *d_out, int img_w,
                                        int img_h, int img_stride, const uint8_t *d_se, int se_w, int se_h,
                                        int anchor_x, int anchor_y, dim3 block_dim, cudaStream_t stream);
CUDA_OPS_API void morphBlackHat_u8_masked(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_close, uint8_t *d_out,
                                          int img_w, int img_h, int img_stride, const uint8_t *d_se, int se_w, int se_h,
                                          int anchor_x, int anchor_y, dim3 block_dim, cudaStream_t stream);

// -------------------- 统一的形态学操作接口 --------------------
CUDA_OPS_API void morphologyEx(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w,
                               int img_h, int img_stride, const int op, const uint8_t *d_se, int se_w, int se_h,
                               int anchor_x, int anchor_y, dim3 block_dim, cudaStream_t stream);

} // namespace free_kick::cuda::ops::v1