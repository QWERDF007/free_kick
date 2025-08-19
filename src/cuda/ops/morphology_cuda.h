#pragma once

#include "opsdef.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace free_kick::cuda::ops {

// -------------------- Host 端接口声明 --------------------
// 通用模板声明（仅声明，不在头文件中定义以避免在非 CUDA 翻译单元中看到 <<< >>> 语法）
template<typename T>
void morphDilate(const T *d_in, T *d_out, int img_w, int img_h, int img_stride, int radius, dim3 block_dim,
                 cudaStream_t stream);

template<typename T>
void morphErode(const T *d_in, T *d_out, int img_w, int img_h, int img_stride, int radius, dim3 block_dim,
                cudaStream_t stream);

template<typename T>
void morphOpen(const T *d_in, T *d_tmp, T *d_out, int img_w, int img_h, int img_stride, int radius, dim3 block_dim,
               cudaStream_t stream);

template<typename T>
void morphClose(const T *d_in, T *d_tmp, T *d_out, int img_w, int img_h, int img_stride, int radius, dim3 block_dim,
                cudaStream_t stream);

template<typename T>
void morphTopHat(const T *d_in, T *d_tmp, T *d_open, T *d_out, int img_w, int img_h, int img_stride, int radius,
                 dim3 block_dim, cudaStream_t stream);

template<typename T>
void morphBlackHat(const T *d_in, T *d_tmp, T *d_close, T *d_out, int img_w, int img_h, int img_stride, int radius,
                   dim3 block_dim, cudaStream_t stream);

// -------------------- uint8_t 专用实现声明 --------------------
// 在 .cu 中提供定义，这里只声明
CUDA_OPS_API void morphDilate_u8(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride, int radius,
                                 dim3 block_dim, cudaStream_t stream);
CUDA_OPS_API void morphErode_u8(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride, int radius,
                                dim3 block_dim, cudaStream_t stream);
CUDA_OPS_API void morphOpen_u8(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_out, int img_w, int img_h,
                               int img_stride, int radius, dim3 block_dim, cudaStream_t stream);
CUDA_OPS_API void morphClose_u8(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_out, int img_w, int img_h,
                                int img_stride, int radius, dim3 block_dim, cudaStream_t stream);
CUDA_OPS_API void morphTopHat_u8(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_open, uint8_t *d_out, int img_w,
                                 int img_h, int img_stride, int radius, dim3 block_dim, cudaStream_t stream);
CUDA_OPS_API void morphBlackHat_u8(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_close, uint8_t *d_out, int img_w,
                                   int img_h, int img_stride, int radius, dim3 block_dim, cudaStream_t stream);

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

// -------------------- 模板对 uint8_t 的特化（转调到 u8 实现） --------------------
template<>
inline void morphDilate<uint8_t>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride, int radius,
                                 dim3 block_dim, cudaStream_t stream)
{
    morphDilate_u8(d_in, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
}

template<>
inline void morphErode<uint8_t>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride, int radius,
                                dim3 block_dim, cudaStream_t stream)
{
    morphErode_u8(d_in, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
}

template<>
inline void morphOpen<uint8_t>(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_out, int img_w, int img_h,
                               int img_stride, int radius, dim3 block_dim, cudaStream_t stream)
{
    morphOpen_u8(d_in, d_tmp, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
}

template<>
inline void morphClose<uint8_t>(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_out, int img_w, int img_h,
                                int img_stride, int radius, dim3 block_dim, cudaStream_t stream)
{
    morphClose_u8(d_in, d_tmp, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
}

template<>
inline void morphTopHat<uint8_t>(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_open, uint8_t *d_out, int img_w,
                                 int img_h, int img_stride, int radius, dim3 block_dim, cudaStream_t stream)
{
    morphTopHat_u8(d_in, d_tmp, d_open, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
}

template<>
inline void morphBlackHat<uint8_t>(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_close, uint8_t *d_out, int img_w,
                                   int img_h, int img_stride, int radius, dim3 block_dim, cudaStream_t stream)
{
    morphBlackHat_u8(d_in, d_tmp, d_close, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
}

} // namespace free_kick::cuda::ops