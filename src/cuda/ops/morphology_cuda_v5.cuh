#pragma once

#include "morph_common.cuh"
#include "morphology_cuda_v4.cuh"

namespace free_kick::cuda::ops {

// v5: 共享内存 + 无分支 clamp + 偏移列表优化 + 分离横向/纵向
template<typename T>
struct v5 : v4<T>
{
    // 主要的operator()函数，组合横向和纵向处理
    // 这个版本需要外部调用方分别调用horizontal_pass和vertical_pass
    // 或者提供分离的偏移列表
    // 这里提供一个简化的实现，假设结构元素可以分离为横向和纵向

    // 注意：这是一个简化实现，实际应用中需要：
    // 1. 预处理结构元素，分离为横向和纵向偏移
    // 2. 使用中间缓冲区进行两次传递
    // 3. 或者提供专门的分离式调用接口

    // 如果结构元素是可分离的（如矩形、十字形等），可以显著降低复杂度
    // 从 O(se_w * se_h) 降低到 O(se_w + se_h)

    // 这里先实现原来的逻辑作为回退
    template<typename Reducer>
    __device__ void operator()(const Reducer &reducer, const T *__restrict__ in, T *__restrict__ out, const int img_w,
                               const int img_h, const int img_stride, const int2 *__restrict__ d_se,
                               const int n_offsets, const int se_w, const int se_h, const int anchor_x,
                               const int anchor_y) const
    {
        v4<T>::operator()(reducer, in, out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
    }

    size_t getSharedMemSize(dim3 block_dim, int se_w, int se_h, int anchor_x, int anchor_y)
    {
        return v4<T>::getSharedMemSize(block_dim, se_w, se_h, anchor_x, anchor_y);
    }

    // --- 水平 pass ---
    template<typename Reducer>
    __device__ void horizontal(const Reducer &reducer, const T *__restrict__ in, T *__restrict__ tmp, const int img_w,
                               const int img_h, const int img_stride, const int se_w, const int anchor_x) const
    {
        extern __shared__ unsigned char smem_u8[];
        T                              *smem = reinterpret_cast<T *>(smem_u8);

        const int left  = anchor_x;
        const int right = se_w - 1 - anchor_x;

        const int tile_w = blockDim.x + left + right;
        const int tile_h = blockDim.y;

        const int block_x = blockIdx.x * blockDim.x;
        const int block_y = blockIdx.y * blockDim.y;

        // --- 将 tile 行读入共享内存 ---
        int gy = block_y + threadIdx.y;
        if (gy < img_h)
        {
            const T *in_row = in + gy * img_stride;
            for (int xx = threadIdx.x; xx < tile_w; xx += blockDim.x)
            {
                int gx                          = clampIndexNoBranch(block_x + xx - left, 0, img_w);
                smem[threadIdx.y * tile_w + xx] = in_row[gx];
            }
        }
        __syncthreads();

        // --- 每个线程计算一个像素的水平结果 ---
        const int x = block_x + threadIdx.x;
        if (x >= img_w || gy >= img_h)
            return;

        T         acc = reducer.init();
        const int sx  = threadIdx.x + left;

        for (int dx = -left; dx <= right; ++dx)
        {
            int idx = threadIdx.y * tile_w + (sx + dx);
            acc     = reducer.reduce(acc, smem[idx]);
        }

        tmp[gy * img_stride + x] = acc;
    }

    // --- 垂直 pass ---
    template<typename Reducer>
    __device__ void vertical(const Reducer &reducer, const T *__restrict__ tmp, T *__restrict__ out, const int img_w,
                             const int img_h, const int img_stride, const int se_h, const int anchor_y) const
    {
        extern __shared__ unsigned char smem_u8[];
        T                              *smem = reinterpret_cast<T *>(smem_u8);

        const int top    = anchor_y;
        const int bottom = se_h - 1 - anchor_y;

        const int tile_w = blockDim.x;
        const int tile_h = blockDim.y + top + bottom;

        const int block_x = blockIdx.x * blockDim.x;
        const int block_y = blockIdx.y * blockDim.y;

        // --- 将 tile 列读入共享内存 ---
        const int x = block_x + threadIdx.x;
        for (int yy = threadIdx.y; yy < tile_h; yy += blockDim.y)
        {
            int gy                          = clampIndexNoBranch(block_y + yy - top, 0, img_h);
            smem[yy * tile_w + threadIdx.x] = tmp[gy * img_stride + x];
        }
        __syncthreads();

        const int y = block_y + threadIdx.y;
        if (x >= img_w || y >= img_h)
            return;

        T         acc = reducer.init();
        const int sy  = threadIdx.y + top;

        for (int dy = -top; dy <= bottom; ++dy)
        {
            int idx = (sy + dy) * tile_w + threadIdx.x;
            acc     = reducer.reduce(acc, smem[idx]);
        }

        out[y * img_stride + x] = acc;
    }
};

} // namespace free_kick::cuda::ops
