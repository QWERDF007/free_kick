#include "common/utility.h"

#include "morphology_unified.cuh"

namespace free_kick::cuda::ops {

// ==================== 模板声明 ====================
template<typename Strategy, typename SEType>
void erode(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride, const SEType *d_se, int n_offsets,
           int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream);

// ==================== ERODE 特化实现 ====================
// DirectAccessStrategy + uint8_t 特化
template<>
void erode<v0::DirectAccessStrategy, uint8_t>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride,
                                              const uint8_t *d_se, int n_offsets, int se_w, int se_h, int anchor_x,
                                              int anchor_y, cudaStream_t stream)
{
    dim3   block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = 0;
    v0::morphKernel<uint8_t, MinReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

// SharedMemoryStrategy + uint8_t 特化
template<>
void erode<v1::SharedMemoryStrategy, uint8_t>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride,
                                              const uint8_t *d_se, int n_offsets, int se_w, int se_h, int anchor_x,
                                              int anchor_y, cudaStream_t stream)
{
    dim3   block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = calcSharedMemSize<uint8_t>(block_dim, se_w, se_h, anchor_x, anchor_y);
    v1::morphKernel<uint8_t, MinReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

// OffsetOptimizedStrategy + int2 特化
template<>
void erode<v2::OffsetOptimizedStrategy, int2>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride,
                                              const int2 *d_se, int n_offsets, int se_w, int se_h, int anchor_x,
                                              int anchor_y, cudaStream_t stream)
{
    dim3   block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = calcSharedMemSize<uint8_t>(block_dim, se_w, se_h, anchor_x, anchor_y);
    v2::morphKernel<uint8_t, MinReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

template<typename Strategy, typename SEType>
void dilate(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h, int img_stride, const SEType *d_se,
            int n_offsets, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream);

// ==================== DILATE 特化实现 ====================
// DirectAccessStrategy + uint8_t 特化
template<>
void dilate<v0::DirectAccessStrategy, uint8_t>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h,
                                               int img_stride, const uint8_t *d_se, int n_offsets, int se_w, int se_h,
                                               int anchor_x, int anchor_y, cudaStream_t stream)
{
    dim3   block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = 0;
    v0::morphKernel<uint8_t, MaxReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

// SharedMemoryStrategy + uint8_t 特化
template<>
void dilate<v1::SharedMemoryStrategy, uint8_t>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h,
                                               int img_stride, const uint8_t *d_se, int n_offsets, int se_w, int se_h,
                                               int anchor_x, int anchor_y, cudaStream_t stream)
{
    dim3   block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = calcSharedMemSize<uint8_t>(block_dim, se_w, se_h, anchor_x, anchor_y);
    v1::morphKernel<uint8_t, MaxReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

// OffsetOptimizedStrategy + int2 特化
template<>
void dilate<v2::OffsetOptimizedStrategy, int2>(const uint8_t *d_in, uint8_t *d_out, int img_w, int img_h,
                                               int img_stride, const int2 *d_se, int n_offsets, int se_w, int se_h,
                                               int anchor_x, int anchor_y, cudaStream_t stream)
{
    dim3   block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3   grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t smem_bytes = calcSharedMemSize<uint8_t>(block_dim, se_w, se_h, anchor_x, anchor_y);
    v2::morphKernel<uint8_t, MaxReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
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
    dim3 block_dim{THREAD_SIZE};
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
    dim3 block_dim{THREAD_SIZE};
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

} // namespace free_kick::cuda::ops