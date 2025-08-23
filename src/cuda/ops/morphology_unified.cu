#include "common/utility.h"

#include "morphology_unified.cuh"

namespace free_kick::cuda::ops::unified {

// ==================== 核函数实现 ====================

// v0 策略：直接全局内存访问
template<typename T, typename Reducer>
__global__ void morphKernelDirectAccess(const T *__restrict__ in, T *__restrict__ out, int img_w, int img_h,
                                        int img_stride, const uint8_t *__restrict__ d_se, int n_offsets, int se_w,
                                        int se_h, int anchor_x, int anchor_y)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= img_w || y >= img_h)
        return;

    Reducer reducer;
    T       acc = reducer.init();

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

// v1 策略：共享内存
template<typename T, typename Reducer>
__global__ void morphKernelSharedMemory(const T *__restrict__ in, T *__restrict__ out, int img_w, int img_h,
                                        int img_stride, const uint8_t *__restrict__ d_se, int n_offsets, int se_w,
                                        int se_h, int anchor_x, int anchor_y)
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

    // 加载共享内存
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

    const int sx = threadIdx.x + left;
    const int sy = threadIdx.y + top;

    for (int ky = 0; ky < se_h; ++ky)
    {
        const int      row    = (sy + (ky - anchor_y)) * tile_w;
        const uint8_t *se_row = d_se + ky * se_w;
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

// v2 策略：偏移优化
template<typename T, typename Reducer>
__global__ void morphKernelOffsetOptimized(const T *__restrict__ in, T *__restrict__ out, int img_w, int img_h,
                                           int img_stride, const int2 *__restrict__ d_se, int n_offsets, int se_w,
                                           int se_h, int anchor_x, int anchor_y)
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

    // 加载共享内存（使用无分支clamp）
    for (int yy = threadIdx.y; yy < tile_h; yy += blockDim.y)
    {
        int      gy     = clampIndexNoBranch(block_y + yy - top, 0, img_h);
        const T *in_row = in + gy * img_stride;

        for (int xx = threadIdx.x; xx < tile_w; xx += blockDim.x)
        {
            int gx                 = clampIndexNoBranch(block_x + xx - left, 0, img_w);
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

    const int sx         = threadIdx.x + left;
    const int sy         = threadIdx.y + top;
    const int row_stride = tile_w;

    // 使用预计算的偏移列表
    for (int i = 0; i < n_offsets; ++i)
    {
        const int ox  = d_se[i].x;
        const int oy  = d_se[i].y;
        const int idx = (sy + oy) * row_stride + (sx + ox);
        acc           = reducer.reduce(acc, smem[idx]);
    }

    out[y * img_stride + x] = acc;
}

// ==================== 模板特化实现 ====================
// 声明通用模板
template<typename Strategy, typename SEType>
void erode(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride, const SEType *d_se, int n_offsets,
           int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream);

// ==================== ERODE 特化实现 ====================
// DirectAccessStrategy + uint8_t 特化
template<>
void erode<DirectAccessStrategy, uint8_t>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride,
                                          const uint8_t *d_se, int n_offsets, int se_w, int se_h, int anchor_x,
                                          int anchor_y, cudaStream_t stream)
{
    dim3   block_dim{32, 16};
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = 0;
    morphKernelDirectAccess<uint8_t, MaxReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

// SharedMemoryStrategy + uint8_t 特化
template<>
void erode<SharedMemoryStrategy, uint8_t>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride,
                                          const uint8_t *d_se, int n_offsets, int se_w, int se_h, int anchor_x,
                                          int anchor_y, cudaStream_t stream)
{
    dim3   block_dim{32, 16};
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = calcSharedMemSize(block_dim, se_w, se_h, anchor_x, anchor_y);
    morphKernelSharedMemory<uint8_t, MaxReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

// OffsetOptimizedStrategy + int2 特化
template<>
void erode<OffsetOptimizedStrategy, int2>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride,
                                          const int2 *d_se, int n_offsets, int se_w, int se_h, int anchor_x,
                                          int anchor_y, cudaStream_t stream)
{
    dim3   block_dim{32, 16};
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = calcSharedMemSize(block_dim, se_w, se_h, anchor_x, anchor_y);
    morphKernelOffsetOptimized<uint8_t, MaxReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

template<typename Strategy, typename SEType>
void dilate(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride, const SEType *d_se,
            int n_offsets, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream);

// ==================== DILATE 特化实现 ====================
// DirectAccessStrategy + uint8_t 特化
template<>
void dilate<DirectAccessStrategy, uint8_t>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride,
                                           const uint8_t *d_se, int n_offsets, int se_w, int se_h, int anchor_x,
                                           int anchor_y, cudaStream_t stream)
{
    dim3   block_dim{32, 16};
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = 0;
    morphKernelDirectAccess<uint8_t, MinReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

// SharedMemoryStrategy + uint8_t 特化
template<>
void dilate<SharedMemoryStrategy, uint8_t>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride,
                                           const uint8_t *d_se, int n_offsets, int se_w, int se_h, int anchor_x,
                                           int anchor_y, cudaStream_t stream)
{
    dim3   block_dim{32, 16};
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = calcSharedMemSize(block_dim, se_w, se_h, anchor_x, anchor_y);
    morphKernelSharedMemory<uint8_t, MinReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

// OffsetOptimizedStrategy + int2 特化
template<>
void dilate<OffsetOptimizedStrategy, int2>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride,
                                           const int2 *d_se, int n_offsets, int se_w, int se_h, int anchor_x,
                                           int anchor_y, cudaStream_t stream)
{
    dim3   block_dim{32, 16};
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = calcSharedMemSize(block_dim, se_w, se_h, anchor_x, anchor_y);
    morphKernelOffsetOptimized<uint8_t, MinReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

template<typename Strategy, typename SEType>
void open(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, int img_w, int img_h, int img_stride, const SEType *d_se,
          int n_offsets, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: Erode
    erode<Strategy, SEType>(d_in, d_tmp, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y,
                            stream);
    // Step 2: Dilate
    dilate<Strategy, SEType>(d_tmp, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y,
                             stream);
}

template<typename Strategy, typename SEType>
void close(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, int img_w, int img_h, int img_stride,
           const SEType *d_se, int n_offsets, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: Dilate
    dilate<Strategy, SEType>(d_in, d_tmp, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y,
                             stream);
    // Step 2: Erode
    erode<Strategy, SEType>(d_tmp, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y,
                            stream);
}

template<typename Strategy, typename SEType>
void tophat(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h, int img_stride,
            const SEType *d_se, int n_offsets, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: Open operation (stored in d_tmp2)
    open<Strategy, SEType>(d_in, d_tmp2, d_tmp, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                           anchor_y, stream);

    // Step 2: Subtract open result from original: out = src - open
    int  total_pixels = img_w * img_h;
    dim3 block_dim{256};
    dim3 grid_dim(divUp(total_pixels, block_dim.x));
    sub_clamp_kernel<uint8_t, 0, 255><<<grid_dim, block_dim, 0, stream>>>(d_in, d_tmp2, d_out, total_pixels);
}

template<typename Strategy, typename SEType>
void blackhat(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h,
              int img_stride, const SEType *d_se, int n_offsets, int se_w, int se_h, int anchor_x, int anchor_y,
              cudaStream_t stream)
{
    // Step 1: Close operation (stored in d_tmp2)
    close<Strategy, SEType>(d_in, d_tmp2, d_tmp, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                            anchor_y, stream);

    // Step 2: Subtract original from close result: out = close - src
    int  total_pixels = img_w * img_h;
    dim3 block_dim{256};
    dim3 grid_dim(divUp(total_pixels, block_dim.x));
    sub_clamp_kernel<uint8_t, 0, 255><<<grid_dim, block_dim, 0, stream>>>(d_tmp2, d_in, d_out, total_pixels);
}

template<typename Strategy, typename SEType>
void morphologyEx(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h,
                  int img_stride, int op, const SEType *d_se, int n_offsets, int se_w, int se_h, int anchor_x,
                  int anchor_y, cudaStream_t stream)
{
    switch (op)
    {
    case cv::MORPH_ERODE:
    {
        erode<Strategy, SEType>(d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y,
                                stream);
        break;
    }
    case cv::MORPH_DILATE:
    {
        dilate<Strategy, SEType>(d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y,
                                 stream);
        break;
    }
    case cv::MORPH_OPEN:
    {
        open<Strategy, SEType>(d_in, d_out, d_tmp, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                               anchor_y, stream);
        break;
    }
    case cv::MORPH_CLOSE:
    {
        close<Strategy, SEType>(d_in, d_out, d_tmp, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                                anchor_y, stream);
        break;
    }
    case cv::MORPH_TOPHAT:
    {
        tophat<Strategy, SEType>(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h,
                                 anchor_x, anchor_y, stream);
        break;
    }
    case cv::MORPH_BLACKHAT:
    {
        blackhat<Strategy, SEType>(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h,
                                   anchor_x, anchor_y, stream);
        break;
    }

    default:
        // Unsupported operation - do nothing or throw error
        break;
    }
}

} // namespace free_kick::cuda::ops::unified