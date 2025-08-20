
#include "common/utility.h"
#include "morphology_cuda_v1.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace free_kick::cuda::ops::v1 {

// -------------------- 设备端辅助 --------------------
__device__ __forceinline__ int clampIndex(int x, int low, int high_exclusive)
{
    if (x < low)
        return low;
    if (x >= high_exclusive)
        return high_exclusive - 1;
    return x;
}

// -------------------- 核函数：共享内存 + Halo --------------------
// -------------------- 核函数：带掩码的通用结构元素 --------------------
// 结构元素 se 为大小 se_w x se_h 的二值掩码，anchor_x/anchor_y 为锚点（通常为中心）。
// 边界采用 replicate（坐标 clamp）。
template<typename T, typename Reducer>
__global__ void morphReduceKernelMasked(const T *__restrict__ in, T *__restrict__ out, int img_w, int img_h,
                                        int img_stride, const uint8_t *__restrict__ se, int se_w, int se_h,
                                        int anchor_x, int anchor_y)
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
        const uint8_t *se_row = se + ky * se_w;
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

// -------------------- Host 端实现（uint8_t 专用，带掩码结构元素） --------------------
void morphDilate_u8_masked(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride,
                           const uint8_t *d_se, int se_w, int se_h, int anchor_x, int anchor_y, dim3 block_dim,
                           cudaStream_t stream)
{
    const int left   = anchor_x;
    const int right  = se_w - 1 - anchor_x;
    const int top    = anchor_y;
    const int bottom = se_h - 1 - anchor_y;

    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = size_t(block_dim.x + left + right) * size_t(block_dim.y + top + bottom) * sizeof(uint8_t);
    morphReduceKernelMasked<uint8_t, MaxReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y);
}

void morphErode_u8_masked(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride,
                          const uint8_t *d_se, int se_w, int se_h, int anchor_x, int anchor_y, dim3 block_dim,
                          cudaStream_t stream)
{
    const int left   = anchor_x;
    const int right  = se_w - 1 - anchor_x;
    const int top    = anchor_y;
    const int bottom = se_h - 1 - anchor_y;

    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = size_t(block_dim.x + left + right) * size_t(block_dim.y + top + bottom) * sizeof(uint8_t);
    morphReduceKernelMasked<uint8_t, MinReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y);
}

void morphOpen_u8_masked(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_out, int img_w, int img_h, int img_stride,
                         const uint8_t *d_se, int se_w, int se_h, int anchor_x, int anchor_y, dim3 block_dim,
                         cudaStream_t stream)
{
    morphErode_u8_masked(d_in, d_tmp, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y, block_dim,
                         stream);
    morphDilate_u8_masked(d_tmp, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y, block_dim,
                          stream);
}

void morphClose_u8_masked(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_out, int img_w, int img_h, int img_stride,
                          const uint8_t *d_se, int se_w, int se_h, int anchor_x, int anchor_y, dim3 block_dim,
                          cudaStream_t stream)
{
    morphDilate_u8_masked(d_in, d_tmp, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y, block_dim,
                          stream);
    morphErode_u8_masked(d_tmp, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y, block_dim,
                         stream);
}

void morphTopHat_u8_masked(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_open, uint8_t *d_out, int img_w, int img_h,
                           int img_stride, const uint8_t *d_se, int se_w, int se_h, int anchor_x, int anchor_y,
                           dim3 block_dim, cudaStream_t stream)
{
    // 顶帽 = 原图 - 开运算（带掩码）
    morphOpen_u8_masked(d_in, d_tmp, d_open, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y, block_dim,
                        stream);

    int numel = img_h * img_stride;
    int thr   = 256;
    int bl    = divUp(numel, thr);
    sub_clamp_kernel<uint8_t, 0, 255><<<bl, thr, 0, stream>>>(d_in, d_open, d_out, numel);
}

void morphBlackHat_u8_masked(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_close, uint8_t *d_out, int img_w,
                             int img_h, int img_stride, const uint8_t *d_se, int se_w, int se_h, int anchor_x,
                             int anchor_y, dim3 block_dim, cudaStream_t stream)
{
    // 黑帽 = 闭运算 - 原图（带掩码）
    morphClose_u8_masked(d_in, d_tmp, d_close, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y,
                         block_dim, stream);

    int numel = img_h * img_stride;
    int thr   = 256;
    int bl    = divUp(numel, thr);
    sub_clamp_kernel<uint8_t, 0, 255><<<bl, thr, 0, stream>>>(d_close, d_in, d_out, numel);
}

typedef void (*MorphFunc)(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_close, uint8_t *d_out, int img_w, int img_h,
                          int img_stride, const uint8_t *d_se, int se_w, int se_h, int anchor_x, int anchor_y,
                          dim3 block_dim, cudaStream_t stream);

// -------------------- 统一的形态学操作接口实现 --------------------
void morphologyEx(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h,
                  int img_stride, const int op, const uint8_t *d_se, int se_w, int se_h, int anchor_x, int anchor_y,
                  dim3 block_dim, cudaStream_t stream)
{
    switch (op)
    {
    case cv::MORPH_DILATE:
        morphDilate_u8_masked(d_in, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y, block_dim,
                              stream);
        break;
    case cv::MORPH_ERODE:
        morphErode_u8_masked(d_in, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y, block_dim,
                             stream);
        break;
    case cv::MORPH_OPEN:
        morphOpen_u8_masked(d_in, d_tmp, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y,
                            block_dim, stream);
        break;
    case cv::MORPH_CLOSE:
        morphClose_u8_masked(d_in, d_tmp, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y,
                             block_dim, stream);
        break;
    case cv::MORPH_TOPHAT:
        morphTopHat_u8_masked(d_in, d_tmp, d_tmp2, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x,
                              anchor_y, block_dim, stream);
        break;
    case cv::MORPH_BLACKHAT:
        morphBlackHat_u8_masked(d_in, d_tmp, d_tmp2, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x,
                                anchor_y, block_dim, stream);
        break;
    default:
        // 不支持的操作，可以抛出异常或返回错误
        break;
    }
}

} // namespace free_kick::cuda::ops::v1