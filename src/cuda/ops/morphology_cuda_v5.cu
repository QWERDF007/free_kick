#include "morphology_cuda_v5.cuh"

namespace free_kick::cuda::ops {

// ==================== 基础操作模板声明 ====================

// 分离式腐蚀操作
template<typename Executor, typename T, int SEShape>
void separable_erode(const T *d_in, T *d_out, T *d_tmp, int img_w, int img_h, int img_stride, int se_w, int se_h,
                     const int anchor_x, const int anchor_y, cudaStream_t stream);

// 分离式膨胀操作
template<typename Executor, typename T, int SEShape>
void separable_dilate(const T *d_in, T *d_out, T *d_tmp, int img_w, int img_h, int img_stride, int se_w, int se_h,
                      const int anchor_x, const int anchor_y, cudaStream_t stream);

// ==================== 基础操作特化实现 ====================

// ========== cv::MORPH_RECT 特化 ==========

// 矩形结构元素腐蚀特化 - 使用分离式水平+垂直算法
template<>
void separable_erode<v5<uint8_t>, uint8_t, cv::MORPH_RECT>(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp,
                                                           int img_w, int img_h, int img_stride, int se_w, int se_h,
                                                           const int anchor_x, const int anchor_y, cudaStream_t stream)
{
    v5<uint8_t>         executor;
    MinReducer<uint8_t> reducer;
    dim3                block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3                grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));

    // 第一步：水平腐蚀 (d_in -> d_tmp)
    size_t smem_h = (block_dim.y) * (block_dim.x + se_w - 1) * sizeof(uint8_t);
    separable_horizontal_kernel<v5<uint8_t>, uint8_t, MinReducer<uint8_t>><<<grid_dim, block_dim, smem_h, stream>>>(
        executor, reducer, d_in, d_tmp, img_w, img_h, img_stride, se_w, anchor_x);

    // 第二步：垂直腐蚀 (d_tmp -> d_out)
    size_t smem_v = (block_dim.y + se_h - 1) * (block_dim.x) * sizeof(uint8_t);
    separable_vertical_kernel<v5<uint8_t>, uint8_t, MinReducer<uint8_t>><<<grid_dim, block_dim, smem_v, stream>>>(
        executor, reducer, d_tmp, d_out, img_w, img_h, img_stride, se_h, anchor_y);
}

// 矩形结构元素膨胀特化 - 使用分离式水平+垂直算法
template<>
void separable_dilate<v5<uint8_t>, uint8_t, cv::MORPH_RECT>(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp,
                                                            int img_w, int img_h, int img_stride, int se_w, int se_h,
                                                            const int anchor_x, const int anchor_y, cudaStream_t stream)
{
    v5<uint8_t>         executor;
    MaxReducer<uint8_t> reducer;
    dim3                block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3                grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));

    // 第一步：水平膨胀 (d_in -> d_tmp)
    size_t smem_h = (block_dim.y) * (block_dim.x + se_w - 1) * sizeof(uint8_t);
    separable_horizontal_kernel<v5<uint8_t>, uint8_t, MaxReducer<uint8_t>><<<grid_dim, block_dim, smem_h, stream>>>(
        executor, reducer, d_in, d_tmp, img_w, img_h, img_stride, se_w, anchor_x);

    // 第二步：垂直膨胀 (d_tmp -> d_out)
    size_t smem_v = (block_dim.y + se_h - 1) * (block_dim.x) * sizeof(uint8_t);
    separable_vertical_kernel<v5<uint8_t>, uint8_t, MaxReducer<uint8_t>><<<grid_dim, block_dim, smem_v, stream>>>(
        executor, reducer, d_tmp, d_out, img_w, img_h, img_stride, se_h, anchor_y);
}

// ========== cv::MORPH_CROSS 特化 ==========

// 十字形结构元素腐蚀特化 - 使用cross算法
template<>
void separable_erode<v5<uint8_t>, uint8_t, cv::MORPH_CROSS>(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp,
                                                            int img_w, int img_h, int img_stride, int se_w, int se_h,
                                                            const int anchor_x, const int anchor_y, cudaStream_t stream)
{
    v5<uint8_t>         executor;
    MinReducer<uint8_t> reducer;
    dim3                block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3                grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));

    // 十字形腐蚀使用 cross kernel
    size_t smem_bytes
        = ((block_dim.y) * (block_dim.x + se_w - 1) + (block_dim.y + se_h - 1) * (block_dim.x)) * sizeof(uint8_t);
    separable_cross_kernel<v5<uint8_t>, uint8_t, MinReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        executor, reducer, d_in, d_tmp, d_out, d_out, img_w, img_h, img_stride, se_w, se_h, anchor_x, anchor_y);
}

// 十字形结构元素膨胀特化 - 使用cross算法
template<>
void separable_dilate<v5<uint8_t>, uint8_t, cv::MORPH_CROSS>(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp,
                                                             int img_w, int img_h, int img_stride, int se_w, int se_h,
                                                             const int anchor_x, const int anchor_y,
                                                             cudaStream_t stream)
{
    v5<uint8_t>         executor;
    MaxReducer<uint8_t> reducer;
    dim3                block_dim{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3                grid_dim(divUp(img_w, block_dim.x), divUp(img_h, block_dim.y));

    // 十字形膨胀使用 cross kernel
    size_t smem_bytes
        = ((block_dim.y) * (block_dim.x + se_w - 1) + (block_dim.y + se_h - 1) * (block_dim.x)) * sizeof(uint8_t);
    separable_cross_kernel<v5<uint8_t>, uint8_t, MaxReducer<uint8_t>><<<grid_dim, block_dim, smem_bytes, stream>>>(
        executor, reducer, d_in, d_tmp, d_out, d_out, img_w, img_h, img_stride, se_w, se_h, anchor_x, anchor_y);
}

// ==================== 复合操作通用模板 ====================

// 通用分离式开运算 - 调用特化后的基础操作
template<typename Executor, int SEShape>
void separable_open(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h,
                    int img_stride, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: 分离式腐蚀 (d_in -> d_tmp2)
    separable_erode<Executor, uint8_t, SEShape>(d_in, d_tmp2, d_tmp, img_w, img_h, img_stride, se_w, se_h, anchor_x,
                                                anchor_y, stream);
    // Step 2: 分离式膨胀 (d_tmp2 -> d_out)
    separable_dilate<Executor, uint8_t, SEShape>(d_tmp2, d_out, d_tmp, img_w, img_h, img_stride, se_w, se_h, anchor_x,
                                                 anchor_y, stream);
}

// 通用分离式闭运算 - 调用特化后的基础操作
template<typename Executor, int SEShape>
void separable_close(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h,
                     int img_stride, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: 分离式膨胀 (d_in -> d_tmp2)
    separable_dilate<Executor, uint8_t, SEShape>(d_in, d_tmp2, d_tmp, img_w, img_h, img_stride, se_w, se_h, anchor_x,
                                                 anchor_y, stream);
    // Step 2: 分离式腐蚀 (d_tmp2 -> d_out)
    separable_erode<Executor, uint8_t, SEShape>(d_tmp2, d_out, d_tmp, img_w, img_h, img_stride, se_w, se_h, anchor_x,
                                                anchor_y, stream);
}

// 通用分离式顶帽运算 - 调用特化后的复合操作
template<typename Executor, int SEShape>
void separable_tophat(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h,
                      int img_stride, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: 分离式开操作 (stored in d_tmp2)
    separable_open<Executor, SEShape>(d_in, d_tmp2, d_tmp, d_out, img_w, img_h, img_stride, se_w, se_h, anchor_x,
                                      anchor_y, stream);
    // Step 2: Subtract open result from original: out = src - open
    int  total_pixels = img_w * img_h;
    dim3 block_dim{THREAD_SIZE};
    dim3 grid_dim(divUp(total_pixels, block_dim.x));
    sub_clamp_kernel<uint8_t, 0, 255><<<grid_dim, block_dim, 0, stream>>>(d_in, d_tmp2, d_out, total_pixels);
}

// 通用分离式黑帽运算 - 调用特化后的复合操作
template<typename Executor, int SEShape>
void separable_blackhat(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, int img_w, int img_h,
                        int img_stride, int se_w, int se_h, int anchor_x, int anchor_y, cudaStream_t stream)
{
    // Step 1: 分离式闭操作 (stored in d_tmp2)
    separable_close<Executor, SEShape>(d_in, d_tmp2, d_tmp, d_out, img_w, img_h, img_stride, se_w, se_h, anchor_x,
                                       anchor_y, stream);
    // Step 2: Subtract original from close result: out = close - src
    int  total_pixels = img_w * img_h;
    dim3 block_dim{THREAD_SIZE};
    dim3 grid_dim(divUp(total_pixels, block_dim.x));
    sub_clamp_kernel<uint8_t, 0, 255><<<grid_dim, block_dim, 0, stream>>>(d_tmp2, d_in, d_out, total_pixels);
}

template<typename Executor, int SEShape>
void morphologyEx(const uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2, const int img_w,
                  const int img_h, const int img_stride, const int op, const int se_w, const int se_h,
                  const int anchor_x, const int anchor_y, cudaStream_t stream)
{
    switch (op)
    {
    case cv::MORPH_ERODE:
    {
        separable_erode<v5<uint8_t>, uint8_t, SEShape>(d_in, d_out, d_tmp, img_w, img_h, img_stride, se_w, se_h,
                                                       anchor_x, anchor_y, stream);
        break;
    }
    case cv::MORPH_DILATE:
    {
        separable_dilate<v5<uint8_t>, uint8_t, SEShape>(d_in, d_out, d_tmp, img_w, img_h, img_stride, se_w, se_h,
                                                        anchor_x, anchor_y, stream);
        break;
    }
    case cv::MORPH_OPEN:
    {
        separable_open<v5<uint8_t>, SEShape>(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, se_w, se_h, anchor_x,
                                             anchor_y, stream);
        break;
    }
    case cv::MORPH_CLOSE:
    {
        separable_close<v5<uint8_t>, SEShape>(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, se_w, se_h,
                                              anchor_x, anchor_y, stream);
        break;
    }
    case cv::MORPH_TOPHAT:
    {
        separable_tophat<v5<uint8_t>, SEShape>(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, se_w, se_h,
                                               anchor_x, anchor_y, stream);
        break;
    }
    case cv::MORPH_BLACKHAT:
    {
        separable_blackhat<v5<uint8_t>, SEShape>(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, se_w, se_h,
                                                 anchor_x, anchor_y, stream);
        break;
    }
    default:
        // 不支持的操作
        break;
    }
}

} // namespace free_kick::cuda::ops
