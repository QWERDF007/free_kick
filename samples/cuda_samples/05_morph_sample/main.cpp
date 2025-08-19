#include "common/utility.h"
#include "morphology_cuda.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace ops = free_kick::cuda::ops;

// 辅助函数：执行形态学操作并与OpenCV对比
void performMorphologyTest(const cv::Mat &img, uint8_t *d_in, uint8_t *d_out, uint8_t *d_tmp, uint8_t *d_tmp2,
                           int img_w, int img_h, int img_stride, const cv::Mat &se, uint8_t *d_se, int se_w, int se_h,
                           int anchor_x, int anchor_y, dim3 block_dim, cudaStream_t stream, cudaEvent_t ev_start,
                           cudaEvent_t ev_stop, const int morph_op, const std::string &op_name, const char *output_dir)
{
    size_t bytes = size_t(img_stride) * img_h * sizeof(uint8_t);

    // CUDA 操作
    float  cuda_ms = 0.0f;
    double wall_ms = 0.0;
    auto   t0      = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    ops::morphologyEx(d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, morph_op, d_se, se_w, se_h, anchor_x,
                      anchor_y, block_dim, stream);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    auto t1 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, ev_start, ev_stop));
    wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cv::Mat out_cuda(img_h, img_w, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(out_cuda.data, d_out, bytes, cudaMemcpyDeviceToHost));

    // OpenCV 操作
    cv::Mat out_cv;
    auto    t0_cv = std::chrono::high_resolution_clock::now();
    cv::morphologyEx(img, out_cv, morph_op, se, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
    auto   t1_cv = std::chrono::high_resolution_clock::now();
    double cv_ms = std::chrono::duration<double, std::milli>(t1_cv - t0_cv).count();

    // 对比结果
    cv::Mat diff;
    cv::absdiff(out_cuda, out_cv, diff);
    double minv = 0.0, maxv = 0.0;
    cv::minMaxLoc(diff, &minv, &maxv);
    int nz = cv::countNonZero(diff);

    // 保存结果
    std::string base = std::string(output_dir) + "/" + op_name;
    cv::imwrite(base + "_cuda.png", out_cuda);
    cv::imwrite(base + "_cv.png", out_cv);

    std::cout << op_name << ": cuda=" << std::fixed << std::setprecision(3) << cuda_ms
              << " ms (event), wall=" << wall_ms << " ms, opencv=" << cv_ms
              << " ms, max_abs_diff=" << std::setprecision(0) << maxv << ", nonzero=" << nz << std::endl;
}

// -------------------- 示例入口 --------------------
int main(int argc, char **argv)
{
    const char *input_path = (argc > 1) ? argv[1] : "input.png";
    const char *output_dir = (argc > 2) ? argv[2] : "./out";
    int         radius     = (argc > 3) ? std::max(0, atoi(argv[3])) : 2; // 结构元素半径
    dim3        block_dim(32, 16);                                        // 可根据显卡调优

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 读入灰度图
    cv::Mat img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        fprintf(stderr, "Failed to load image: %s\n", input_path);
        return 1;
    }
    int img_w      = img.cols;
    int img_h      = img.rows;
    int img_stride = img.cols; // 我们按紧凑行存储（无 padding）

    // 使用 OpenCV 的结构元素（矩形，与 CUDA 方形半径匹配），并拷贝到 GPU 端用于“带掩码”的 CUDA 算子
    int      ksz      = 2 * radius + 1;
    cv::Mat  se       = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksz, ksz));
    int      se_w     = se.cols;
    int      se_h     = se.rows;
    int      anchor_x = se_w / 2;
    int      anchor_y = se_h / 2;
    uint8_t *d_se     = nullptr;
    size_t   se_bytes = static_cast<size_t>(se_w * se_h) * sizeof(uint8_t);
    CUDA_CHECK(cudaMalloc(&d_se, se_bytes));
    CUDA_CHECK(cudaMemcpy(d_se, se.data, se_bytes, cudaMemcpyHostToDevice));

    std::cout << "se.size: " << se.size << std::endl;

    // 分配 GPU 内存
    size_t   bytes = size_t(img_stride) * img_h * sizeof(uint8_t);
    uint8_t *d_in = nullptr, *d_out = nullptr, *d_tmp = nullptr, *d_tmp2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp2, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, img.data, bytes, cudaMemcpyHostToDevice));

    // CUDA 事件计时器
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // 依次对比六种操作（CUDA vs OpenCV）
    std::vector<std::pair<int, std::string>> ops = {
        {   cv::MORPH_ERODE,    "erode"},
        {  cv::MORPH_DILATE,   "dilate"},
        {    cv::MORPH_OPEN,     "open"},
        {   cv::MORPH_CLOSE,    "close"},

        {  cv::MORPH_TOPHAT,   "tophat"},
        {cv::MORPH_BLACKHAT, "blackhat"},
    };
    for (const auto &[op, name] : ops)
    {
        performMorphologyTest(img, d_in, d_out, d_tmp, d_tmp2, img_w, img_h, img_stride, se, d_se, se_w, se_h, anchor_x,
                              anchor_y, block_dim, stream, ev_start, ev_stop, op, name, output_dir);
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_tmp2));
    CUDA_CHECK(cudaFree(d_se));
    printf("Done.\n");
    return 0;
}
