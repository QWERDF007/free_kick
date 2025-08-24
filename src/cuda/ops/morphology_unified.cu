#include "common/utility.h"

#include "morphology_unified.cuh"

namespace free_kick::cuda::ops {

template<typename Executor, typename T, typename SEType, typename Reducer>
__global__ void kernel(const Executor &executor, const Reducer &reducer, const T *__restrict__ in, T *__restrict__ out,
                       const int img_w, const int img_h, const int img_stride, const SEType *__restrict__ d_se,
                       const int n_offsets, const int se_w, const int se_h, const int anchor_x, const int anchor_y)
{
    executor(reducer, in, out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

// ==================== 模板声明 ====================
template<typename T, typename Executor, typename SEType>
void erode(const T *d_in, T *d_out, int img_w, int img_h, int img_stride, const SEType *d_se, int n_offsets, int se_w,
           int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    Executor      executor;
    MinReducer<T> reducer;
    dim3          block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3          grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t        smem_bytes = executor.calcSharedMemSize(block_dim, se_w, se_h, anchor_x, anchor_y);
    kernel<Executor, T, SEType, MinReducer<T>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        executor, reducer, d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

template<typename T, typename Executor, typename SEType>
void dilate(const T *d_in, T *d_out, int img_w, int img_h, int img_stride, const SEType *d_se, int n_offsets, int se_w,
            int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    Executor      executor;
    MaxReducer<T> reducer;
    dim3          block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3          grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));
    size_t        smem_bytes = executor.calcSharedMemSize(block_dim, se_w, se_h, anchor_x, anchor_y);
    kernel<Executor, T, SEType, MaxReducer<T>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        executor, reducer, d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

template<typename Executor, typename SEType>
void open(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, int img_w, int img_h, int img_stride, const SEType *d_se,
          int n_offsets, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: Erode
    erode<uint8_t, Executor, SEType>(d_in, d_tmp, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                                     anchor_y, stream);
    // Step 2: Dilate
    dilate<uint8_t, Executor, SEType>(d_tmp, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                                      anchor_y, stream);
}

template<typename Executor, typename SEType>
void close(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, int img_w, int img_h, int img_stride,
           const SEType *d_se, int n_offsets, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: Dilate
    dilate<uint8_t, Executor, SEType>(d_in, d_tmp, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                                      anchor_y, stream);
    // Step 2: Erode
    erode<uint8_t, Executor, SEType>(d_tmp, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                                     anchor_y, stream);
}

template<typename Executor, typename SEType>
void tophat(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h, int img_stride,
            const SEType *d_se, int n_offsets, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: Open operation (stored in d_tmp2)
    open<Executor, SEType>(d_in, d_tmp2, d_tmp, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                           anchor_y, stream);
    // Step 2: Subtract open result from original: out = src - open
    int  total_pixels = img_w * img_h;
    dim3 block_dim{THREAD_SIZE};
    dim3 grid_dim(divUp(total_pixels, block_dim.x));
    sub_clamp_kernel<uint8_t, 0, 255><<<grid_dim, block_dim, 0, stream>>>(d_in, d_tmp2, d_out, total_pixels);
}

template<typename Executor, typename SEType>
void blackhat(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h,
              int img_stride, const SEType *d_se, int n_offsets, int se_w, int se_h, int anchor_x, int anchor_y,
              cudaStream_t stream)
{
    // Step 1: Close operation (stored in d_tmp2)
    close<Executor, SEType>(d_in, d_tmp2, d_tmp, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                            anchor_y, stream);
    // Step 2: Subtract original from close result: out = close - src
    int  total_pixels = img_w * img_h;
    dim3 block_dim{THREAD_SIZE};
    dim3 grid_dim(divUp(total_pixels, block_dim.x));
    sub_clamp_kernel<uint8_t, 0, 255><<<grid_dim, block_dim, 0, stream>>>(d_tmp2, d_in, d_out, total_pixels);
}

template<typename Executor, typename SEType>
void morphologyEx(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, const int img_w,
                  const int img_h, const int img_stride, const int op, const SEType *d_se, const int n_offsets,
                  const int se_w, const int se_h, const int anchor_x, const int anchor_y, cudaStream_t stream)
{
    switch (op)
    {
    case cv::MORPH_ERODE:
    {
        erode<uint8_t, Executor, SEType>(d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                                         anchor_y, stream);
        break;
    }
    case cv::MORPH_DILATE:
    {
        dilate<uint8_t, Executor, SEType>(d_in, d_out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                                          anchor_y, stream);
        break;
    }
    case cv::MORPH_OPEN:
    {
        open<Executor, SEType>(d_in, d_out, d_tmp, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                               anchor_y, stream);
        break;
    }
    case cv::MORPH_CLOSE:
    {
        close<Executor, SEType>(d_in, d_out, d_tmp, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x,
                                anchor_y, stream);
        break;
    }
    case cv::MORPH_TOPHAT:
    {
        tophat<Executor, SEType>(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h,
                                 anchor_x, anchor_y, stream);
        break;
    }
    case cv::MORPH_BLACKHAT:
    {
        blackhat<Executor, SEType>(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h,
                                   anchor_x, anchor_y, stream);
        break;
    }

    default:
        // Unsupported operation - do nothing or throw error
        break;
    }
}

} // namespace free_kick::cuda::ops