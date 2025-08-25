#include "common/utility.h"

#include "morphology_unified.cuh"

namespace free_kick::cuda::ops {

template<typename Executor, typename T, typename SEType, typename Reducer>
__global__ void kernel(const Executor &executor, const Reducer &reducer, const T *__restrict__ in, T *__restrict__ out,
                       const int img_w, const int img_h, const int img_stride, const SEType *__restrict__ d_se,
                       const int n_offsets, const int se_w, const int se_h, const int anchor_x, const int anchor_y)
{
    // 调用 Executor 中的 __device__ void operator()
    executor(reducer, in, out, img_w, img_h, img_stride, d_se, n_offsets, se_w, se_h, anchor_x, anchor_y);
}

// ==================== 分离式形态学操作（v5专用）====================
template<typename Executor, typename T, typename Reducer>
__global__ void separable_horizontal_kernel(const Executor &executor, const Reducer &reducer, const T *__restrict__ in,
                                            T *__restrict__ tmp, const int img_w, const int img_h, const int img_stride,
                                            const int se_w, const int anchor_x)
{
    executor.horizontal(reducer, in, tmp, img_w, img_h, img_stride, se_w, anchor_x);
}

template<typename Executor, typename T, typename Reducer>
__global__ void separable_vertical_kernel(const Executor &executor, const Reducer &reducer, const T *__restrict__ tmp,
                                          T *__restrict__ out, const int img_w, const int img_h, const int img_stride,
                                          const int se_h, const int anchor_y)
{
    executor.vertical(reducer, tmp, out, img_w, img_h, img_stride, se_h, anchor_y);
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
    size_t        smem_bytes = executor.getSharedMemSize(block_dim, se_w, se_h, anchor_x, anchor_y);
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
    size_t        smem_bytes = executor.getSharedMemSize(block_dim, se_w, se_h, anchor_x, anchor_y);
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

// ==================== 分离式形态学操作（v5专用）====================

// 分离式腐蚀操作
template<typename Executor, typename T>
void separable_erode(const T *d_in, T *d_out, T *d_tmp, int img_w, int img_h, int img_stride, int se_w, int se_h,
                     const int anchor_x, const int anchor_y, cudaStream_t stream)
{
    Executor      executor;
    MinReducer<T> reducer;
    dim3          block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3          grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));

    // 第一步：水平腐蚀 (d_in -> d_tmp)
    size_t smem_h = (block_dim.y) * (block_dim.x + se_w - 1) * sizeof(T);
    separable_horizontal_kernel<Executor, T, MinReducer<T>><<<grid_dim, block_dim, smem_h, stream>>>(
        executor, reducer, d_in, d_tmp, img_w, img_h, img_stride, se_w, anchor_x);

    // 第二步：垂直腐蚀 (d_tmp -> d_out)
    size_t smem_v = (block_dim.y + se_h - 1) * (block_dim.x) * sizeof(T);
    separable_vertical_kernel<Executor, T, MinReducer<T>><<<grid_dim, block_dim, smem_v, stream>>>(
        executor, reducer, d_tmp, d_out, img_w, img_h, img_stride, se_h, anchor_y);
}

// 分离式膨胀操作
template<typename Executor, typename T>
void separable_dilate(const T *d_in, T *d_out, T *d_tmp, int img_w, int img_h, int img_stride, int se_w, int se_h,
                      const int anchor_x, const int anchor_y, cudaStream_t stream)
{
    Executor      executor;
    MaxReducer<T> reducer;
    dim3          block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3          grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));

    // 第一步：水平膨胀 (d_in -> d_tmp)
    size_t smem_h = (block_dim.y) * (block_dim.x + se_w - 1) * sizeof(T);
    separable_horizontal_kernel<Executor, T, MaxReducer<T>><<<grid_dim, block_dim, smem_h, stream>>>(
        executor, reducer, d_in, d_tmp, img_w, img_h, img_stride, se_w, anchor_x);

    // 第二步：垂直膨胀 (d_tmp -> d_out)
    size_t smem_v = (block_dim.y + se_h - 1) * (block_dim.x) * sizeof(T);
    separable_vertical_kernel<Executor, T, MaxReducer<T>><<<grid_dim, block_dim, smem_v, stream>>>(
        executor, reducer, d_tmp, d_out, img_w, img_h, img_stride, se_h, anchor_y);
}

// 分离式开操作
template<typename Executor>
void separable_open(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h,
                    int img_stride, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: 分离式腐蚀 (d_in -> d_tmp2)
    separable_erode<Executor, uint8_t>(d_in, d_tmp2, d_tmp, img_w, img_h, img_stride, se_w, se_h, anchor_x, anchor_y,
                                       stream);
    // Step 2: 分离式膨胀 (d_tmp2 -> d_out)
    separable_dilate<Executor, uint8_t>(d_tmp2, d_out, d_tmp, img_w, img_h, img_stride, se_w, se_h, anchor_x, anchor_y,
                                        stream);
}

// 分离式闭操作
template<typename Executor>
void separable_close(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h,
                     int img_stride, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: 分离式膨胀 (d_in -> d_tmp2)
    separable_dilate<Executor, uint8_t>(d_in, d_tmp2, d_tmp, img_w, img_h, img_stride, se_w, se_h, anchor_x, anchor_y,
                                        stream);
    // Step 2: 分离式腐蚀 (d_tmp2 -> d_out)
    separable_erode<Executor, uint8_t>(d_tmp2, d_out, d_tmp, img_w, img_h, img_stride, se_w, se_h, anchor_x, anchor_y,
                                       stream);
}

// 分离式顶帽操作
template<typename Executor>
void separable_tophat(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h,
                      int img_stride, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: 分离式开操作 (stored in d_tmp2)
    separable_open<Executor>(d_in, d_tmp2, d_tmp, d_out, img_w, img_h, img_stride, se_w, se_h, anchor_x, anchor_y,
                             stream);
    // Step 2: Subtract open result from original: out = src - open
    int  total_pixels = img_w * img_h;
    dim3 block_dim{THREAD_SIZE};
    dim3 grid_dim(divUp(total_pixels, block_dim.x));
    sub_clamp_kernel<uint8_t, 0, 255><<<grid_dim, block_dim, 0, stream>>>(d_in, d_tmp2, d_out, total_pixels);
}

// 分离式黑帽操作
template<typename Executor>
void separable_blackhat(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h,
                        int img_stride, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: 分离式闭操作 (stored in d_tmp2)
    separable_close<Executor>(d_in, d_tmp2, d_tmp, d_out, img_w, img_h, img_stride, se_w, se_h, anchor_x, anchor_y,
                              stream);
    // Step 2: Subtract original from close result: out = close - src
    int  total_pixels = img_w * img_h;
    dim3 block_dim{THREAD_SIZE};
    dim3 grid_dim(divUp(total_pixels, block_dim.x));
    sub_clamp_kernel<uint8_t, 0, 255><<<grid_dim, block_dim, 0, stream>>>(d_tmp2, d_in, d_out, total_pixels);
}

// ==================== 分离式复合形态学操作（v5专用）====================

// v5 特化版本的具体实现
template<>
void morphologyEx<v5<uint8_t>, int2>(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2,
                                     const int img_w, const int img_h, const int img_stride, const int op,
                                     const int2 *d_se, const int n_offsets, const int se_w, const int se_h,
                                     const int anchor_x, const int anchor_y, cudaStream_t stream)
{
    if (d_se == nullptr)
    {
        switch (op)
        {
        case cv::MORPH_ERODE:
        {
            separable_erode<v5<uint8_t>, uint8_t>(d_in, d_out, d_tmp, img_w, img_h, img_stride, se_w, se_h, anchor_x,
                                                  anchor_y, stream);
            break;
        }
        case cv::MORPH_DILATE:
        {
            separable_dilate<v5<uint8_t>, uint8_t>(d_in, d_out, d_tmp, img_w, img_h, img_stride, se_w, se_h, anchor_x,
                                                   anchor_y, stream);
            break;
        }
        case cv::MORPH_OPEN:
        {
            separable_open<v5<uint8_t>>(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, se_w, se_h, anchor_x,
                                        anchor_y, stream);
            break;
        }
        case cv::MORPH_CLOSE:
        {
            separable_close<v5<uint8_t>>(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, se_w, se_h, anchor_x,
                                         anchor_y, stream);
            break;
        }
        case cv::MORPH_TOPHAT:
        {
            separable_tophat<v5<uint8_t>>(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, se_w, se_h, anchor_x,
                                          anchor_y, stream);
            break;
        }
        case cv::MORPH_BLACKHAT:
        {
            separable_blackhat<v5<uint8_t>>(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, se_w, se_h, anchor_x,
                                            anchor_y, stream);
            break;
        }
        default:
            // 不支持的操作
            break;
        }
    }
    else
    {
        // 运行 v4 的特化版本
        morphologyEx<v4<uint8_t>, int2>(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, op, d_se, n_offsets, se_w,
                                        se_h, anchor_x, anchor_y, stream);
    }
}

} // namespace free_kick::cuda::ops