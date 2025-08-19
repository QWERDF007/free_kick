
#include "common/utility.h"
#include "morphology_cuda.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace free_kick::cuda::ops {

// -------------------- 通用工具 --------------------
inline static int divUp(int a, int b)
{
    return (a + b - 1) / b;
}

// -------------------- 设备端辅助 --------------------
__device__ __forceinline__ int clampIndex(int x, int low, int high_exclusive)
{
    if (x < low)
        return low;
    if (x >= high_exclusive)
        return high_exclusive - 1;
    return x;
}

template<typename T>
struct MaxReducer
{
    __device__ __forceinline__ T init() const
    {
        return std::numeric_limits<T>::min();
    }

    __device__ __forceinline__ T reduce(T a, T b) const
    {
        return a > b ? a : b;
    }
};

template<typename T>
struct MinReducer
{
    __device__ __forceinline__ T init() const
    {
        return std::numeric_limits<T>::max();
    }

    __device__ __forceinline__ T reduce(T a, T b) const
    {
        return a < b ? a : b;
    }
};

// -------------------- 核函数：共享内存 + Halo --------------------
// 支持任意半径的方形结构元素；边界采用 replicate（坐标 clamp）
template<typename T, typename Reducer>
__global__ void morphReduceKernel(const T *__restrict__ in, T *__restrict__ out, int img_w, int img_h,
                                  int img_stride, // stride: 每行元素数量（单位: 像素/元素）
                                  int radius)
{
    extern __shared__ unsigned char smem_u8[];
    T                              *smem = reinterpret_cast<T *>(smem_u8);

    const int tile_w = blockDim.x + 2 * radius;
    const int tile_h = blockDim.y + 2 * radius;

    const int block_x = blockIdx.x * blockDim.x;
    const int block_y = blockIdx.y * blockDim.y;

    // 线程在共享内存中的线性加载（分块遍历）：
    for (int yy = threadIdx.y; yy < tile_h; yy += blockDim.y)
    {
        int      gy     = clampIndex(block_y + yy - radius, 0, img_h);
        const T *in_row = in + gy * img_stride;

        for (int xx = threadIdx.x; xx < tile_w; xx += blockDim.x)
        {
            int gx                 = clampIndex(block_x + xx - radius, 0, img_w);
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

    // 以共享内存为中心进行 (2r+1)^2 的 reduce
    const int sx = threadIdx.x + radius;
    const int sy = threadIdx.y + radius;

    for (int dy = -radius; dy <= radius; ++dy)
    {
        const int row = (sy + dy) * tile_w;
        for (int dx = -radius; dx <= radius; ++dx)
        {
            acc = reducer.reduce(acc, smem[row + (sx + dx)]);
        }
    }

    out[y * img_stride + x] = acc;
}

// -------------------- 核函数：带掩码的通用结构元素 --------------------
// 结构元素 se 为大小 se_w x se_h 的二值掩码，anchor_x/anchor_y 为锚点（通常为中心）。
// 边界采用 replicate（坐标 clamp）。
template<typename T, typename Reducer>
__global__ void morphReduceKernelMasked(const T *__restrict__ in, T *__restrict__ out, int img_w, int img_h,
                                        int img_stride, const uint8_t *__restrict__ se, int se_w, int se_h,
                                        int anchor_x, int anchor_y)
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

// -------------------- 简单逐点差值核（带饱和） --------------------
__global__ void sub_clamp_u8_kernel(const uint8_t *a, const uint8_t *b, uint8_t *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int v = int(a[i]) - int(b[i]);
        v     = v < 0 ? 0 : (v > 255 ? 255 : v);
        c[i]  = static_cast<uint8_t>(v);
    }
}

// -------------------- Host 端实现（uint8_t 专用） --------------------
void morphDilate_u8(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride, int radius,
                    dim3 block_dim, cudaStream_t stream)
{
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = size_t(block_dim.x + 2 * radius) * size_t(block_dim.y + 2 * radius) * sizeof(uint8_t);
    morphReduceKernel<uint8_t, MaxReducer<uint8_t>>
        <<<grid_dim, block_dim, smem_bytes, stream>>>(d_in, d_out, img_w, img_h, img_stride, radius);
    CUDA_CHECK(cudaGetLastError());
}

void morphErode_u8(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride, int radius,
                   dim3 block_dim, cudaStream_t stream)
{
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = size_t(block_dim.x + 2 * radius) * size_t(block_dim.y + 2 * radius) * sizeof(uint8_t);
    morphReduceKernel<uint8_t, MinReducer<uint8_t>>
        <<<grid_dim, block_dim, smem_bytes, stream>>>(d_in, d_out, img_w, img_h, img_stride, radius);
    CUDA_CHECK(cudaGetLastError());
}

void morphOpen_u8(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_out, int img_w, int img_h, int img_stride, int radius,
                  dim3 block_dim, cudaStream_t stream)
{
    morphErode_u8(d_in, d_tmp, img_w, img_h, img_stride, radius, block_dim, stream);
    morphDilate_u8(d_tmp, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
}

void morphClose_u8(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_out, int img_w, int img_h, int img_stride,
                   int radius, dim3 block_dim, cudaStream_t stream)
{
    morphDilate_u8(d_in, d_tmp, img_w, img_h, img_stride, radius, block_dim, stream);
    morphErode_u8(d_tmp, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
}

void morphTopHat_u8(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_open, uint8_t *d_out, int img_w, int img_h,
                    int img_stride, int radius, dim3 block_dim, cudaStream_t stream)
{
    // 顶帽 = 原图 - 开运算
    morphOpen_u8(d_in, d_tmp, d_open, img_w, img_h, img_stride, radius, block_dim, stream);

    int numel = img_h * img_stride;
    int thr   = 256;
    int bl    = divUp(numel, thr);
    sub_clamp_u8_kernel<<<bl, thr, 0, stream>>>(d_in, d_open, d_out, numel);
    CUDA_CHECK(cudaGetLastError());
}

void morphBlackHat_u8(const uint8_t *d_in, uint8_t *d_tmp, uint8_t *d_close, uint8_t *d_out, int img_w, int img_h,
                      int img_stride, int radius, dim3 block_dim, cudaStream_t stream)
{
    // 黑帽 = 闭运算 - 原图
    morphClose_u8(d_in, d_tmp, d_close, img_w, img_h, img_stride, radius, block_dim, stream);

    int numel = img_h * img_stride;
    int thr   = 256;
    int bl    = divUp(numel, thr);
    sub_clamp_u8_kernel<<<bl, thr, 0, stream>>>(d_close, d_in, d_out, numel);
    CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaGetLastError());
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
    sub_clamp_u8_kernel<<<bl, thr, 0, stream>>>(d_in, d_open, d_out, numel);
    CUDA_CHECK(cudaGetLastError());
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
    sub_clamp_u8_kernel<<<bl, thr, 0, stream>>>(d_close, d_in, d_out, numel);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace free_kick::cuda::ops