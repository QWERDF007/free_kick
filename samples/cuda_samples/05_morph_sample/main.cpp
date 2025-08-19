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
    // 1) 膨胀
    float  cuda_ms = 0.0f;
    double wall_ms = 0.0;
    double cv_ms   = 0.0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        ops::morphDilate<uint8_t>(d_in, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        auto t1 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, ev_start, ev_stop));
        wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    {
        cv::Mat out_cuda(img_h, img_w, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(out_cuda.data, d_out, bytes, cudaMemcpyDeviceToHost));

        cv::Mat out_cv;
        auto    t0_cv = std::chrono::high_resolution_clock::now();
        cv::morphologyEx(img, out_cv, cv::MORPH_DILATE, se, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        auto t1_cv = std::chrono::high_resolution_clock::now();
        cv_ms      = std::chrono::duration<double, std::milli>(t1_cv - t0_cv).count();

        cv::Mat diff;
        cv::absdiff(out_cuda, out_cv, diff);
        double minv = 0.0, maxv = 0.0;
        cv::minMaxLoc(diff, &minv, &maxv);
        int nz = cv::countNonZero(diff);

        std::string base = std::string(output_dir) + "/dilate";
        cv::imwrite(base + "_cuda.png", out_cuda);
        cv::imwrite(base + "_cv.png", out_cv);
        cv::imwrite(base + "_diff.png", diff);
        printf("dilate: cuda=%.3f ms (event), wall=%.3f ms, opencv=%.3f ms, max_abs_diff=%.0f, nonzero=%d\n", cuda_ms,
               wall_ms, cv_ms, maxv, nz);

        // 带掩码 CUDA（与 OpenCV 的结构元素一致）
        float  cuda_ms_mask = 0.0f;
        double wall_ms_mask = 0.0;
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventRecord(ev_start, stream));
            ops::morphDilate_u8_masked(d_in, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y,
                                       block_dim, stream);
            CUDA_CHECK(cudaEventRecord(ev_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            auto t1 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventElapsedTime(&cuda_ms_mask, ev_start, ev_stop));
            wall_ms_mask = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        cv::Mat out_mask_cuda(img_h, img_w, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(out_mask_cuda.data, d_out, bytes, cudaMemcpyDeviceToHost));

        cv::Mat diff_mask;
        cv::absdiff(out_mask_cuda, out_cv, diff_mask);
        cv::minMaxLoc(diff_mask, &minv, &maxv);
        nz = cv::countNonZero(diff_mask);
        cv::imwrite(base + "_maskcuda.png", out_mask_cuda);
        cv::imwrite(base + "_maskdiff.png", diff_mask);
        printf("dilate(masked): cuda=%.3f ms (event), wall=%.3f ms, max_abs_diff=%.0f, nonzero=%d\n", cuda_ms_mask,
               wall_ms_mask, maxv, nz);
    }

    // 2) 腐蚀
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        ops::morphErode<uint8_t>(d_in, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        auto t1 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, ev_start, ev_stop));
        wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    {
        cv::Mat out_cuda(img_h, img_w, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(out_cuda.data, d_out, bytes, cudaMemcpyDeviceToHost));

        cv::Mat out_cv;
        auto    t0_cv = std::chrono::high_resolution_clock::now();
        cv::morphologyEx(img, out_cv, cv::MORPH_ERODE, se, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        auto t1_cv = std::chrono::high_resolution_clock::now();
        cv_ms      = std::chrono::duration<double, std::milli>(t1_cv - t0_cv).count();

        cv::Mat diff;
        cv::absdiff(out_cuda, out_cv, diff);
        double minv = 0.0, maxv = 0.0;
        cv::minMaxLoc(diff, &minv, &maxv);
        int nz = cv::countNonZero(diff);

        std::string base = std::string(output_dir) + "/erode";
        cv::imwrite(base + "_cuda.png", out_cuda);
        cv::imwrite(base + "_cv.png", out_cv);
        cv::imwrite(base + "_diff.png", diff);
        printf("erode: cuda=%.3f ms (event), wall=%.3f ms, opencv=%.3f ms, max_abs_diff=%.0f, nonzero=%d\n", cuda_ms,
               wall_ms, cv_ms, maxv, nz);

        // 带掩码 CUDA 对比
        float  cuda_ms_mask = 0.0f;
        double wall_ms_mask = 0.0;
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventRecord(ev_start, stream));
            ops::morphErode_u8_masked(d_in, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y,
                                      block_dim, stream);
            CUDA_CHECK(cudaEventRecord(ev_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            auto t1 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventElapsedTime(&cuda_ms_mask, ev_start, ev_stop));
            wall_ms_mask = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        cv::Mat out_mask_cuda(img_h, img_w, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(out_mask_cuda.data, d_out, bytes, cudaMemcpyDeviceToHost));

        cv::Mat diff_mask;
        cv::absdiff(out_mask_cuda, out_cv, diff_mask);
        cv::minMaxLoc(diff_mask, &minv, &maxv);
        nz = cv::countNonZero(diff_mask);
        cv::imwrite(base + "_maskcuda.png", out_mask_cuda);
        cv::imwrite(base + "_maskdiff.png", diff_mask);
        printf("erode(masked): cuda=%.3f ms (event), wall=%.3f ms, max_abs_diff=%.0f, nonzero=%d\n", cuda_ms_mask,
               wall_ms_mask, maxv, nz);
    }

    // 3) 开运算（腐蚀->膨胀）
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        ops::morphOpen<uint8_t>(d_in, d_tmp, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        auto t1 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, ev_start, ev_stop));
        wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    {
        cv::Mat out_cuda(img_h, img_w, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(out_cuda.data, d_out, bytes, cudaMemcpyDeviceToHost));

        cv::Mat out_cv;
        auto    t0_cv = std::chrono::high_resolution_clock::now();
        cv::morphologyEx(img, out_cv, cv::MORPH_OPEN, se, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        auto t1_cv = std::chrono::high_resolution_clock::now();
        cv_ms      = std::chrono::duration<double, std::milli>(t1_cv - t0_cv).count();

        cv::Mat diff;
        cv::absdiff(out_cuda, out_cv, diff);
        double minv = 0.0, maxv = 0.0;
        cv::minMaxLoc(diff, &minv, &maxv);
        int nz = cv::countNonZero(diff);

        std::string base = std::string(output_dir) + "/open";
        cv::imwrite(base + "_cuda.png", out_cuda);
        cv::imwrite(base + "_cv.png", out_cv);
        cv::imwrite(base + "_diff.png", diff);
        printf("open: cuda=%.3f ms (event), wall=%.3f ms, opencv=%.3f ms, max_abs_diff=%.0f, nonzero=%d\n", cuda_ms,
               wall_ms, cv_ms, maxv, nz);

        // 带掩码 CUDA 对比
        float  cuda_ms_mask = 0.0f;
        double wall_ms_mask = 0.0;
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventRecord(ev_start, stream));
            ops::morphOpen_u8_masked(d_in, d_tmp, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x, anchor_y,
                                     block_dim, stream);
            CUDA_CHECK(cudaEventRecord(ev_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            auto t1 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventElapsedTime(&cuda_ms_mask, ev_start, ev_stop));
            wall_ms_mask = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        cv::Mat out_mask_cuda(img_h, img_w, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(out_mask_cuda.data, d_out, bytes, cudaMemcpyDeviceToHost));

        cv::Mat diff_mask;
        cv::absdiff(out_mask_cuda, out_cv, diff_mask);
        cv::minMaxLoc(diff_mask, &minv, &maxv);
        nz = cv::countNonZero(diff_mask);
        cv::imwrite(base + "_maskcuda.png", out_mask_cuda);
        cv::imwrite(base + "_maskdiff.png", diff_mask);
        printf("open(masked): cuda=%.3f ms (event), wall=%.3f ms, max_abs_diff=%.0f, nonzero=%d\n", cuda_ms_mask,
               wall_ms_mask, maxv, nz);
    }

    // 4) 闭运算（膨胀->腐蚀）
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        ops::morphClose<uint8_t>(d_in, d_tmp, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        auto t1 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, ev_start, ev_stop));
        wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    {
        cv::Mat out_cuda(img_h, img_w, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(out_cuda.data, d_out, bytes, cudaMemcpyDeviceToHost));

        cv::Mat out_cv;
        auto    t0_cv = std::chrono::high_resolution_clock::now();
        cv::morphologyEx(img, out_cv, cv::MORPH_CLOSE, se, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        auto t1_cv = std::chrono::high_resolution_clock::now();
        cv_ms      = std::chrono::duration<double, std::milli>(t1_cv - t0_cv).count();

        cv::Mat diff;
        cv::absdiff(out_cuda, out_cv, diff);
        double minv = 0.0, maxv = 0.0;
        cv::minMaxLoc(diff, &minv, &maxv);
        int nz = cv::countNonZero(diff);

        std::string base = std::string(output_dir) + "/close";
        cv::imwrite(base + "_cuda.png", out_cuda);
        cv::imwrite(base + "_cv.png", out_cv);
        cv::imwrite(base + "_diff.png", diff);
        printf("close: cuda=%.3f ms (event), wall=%.3f ms, opencv=%.3f ms, max_abs_diff=%.0f, nonzero=%d\n", cuda_ms,
               wall_ms, cv_ms, maxv, nz);

        // 带掩码 CUDA 对比
        float  cuda_ms_mask = 0.0f;
        double wall_ms_mask = 0.0;
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventRecord(ev_start, stream));
            ops::morphClose_u8_masked(d_in, d_tmp, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x,
                                      anchor_y, block_dim, stream);
            CUDA_CHECK(cudaEventRecord(ev_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            auto t1 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventElapsedTime(&cuda_ms_mask, ev_start, ev_stop));
            wall_ms_mask = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        cv::Mat out_mask_cuda(img_h, img_w, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(out_mask_cuda.data, d_out, bytes, cudaMemcpyDeviceToHost));

        cv::Mat diff_mask;
        cv::absdiff(out_mask_cuda, out_cv, diff_mask);
        cv::minMaxLoc(diff_mask, &minv, &maxv);
        nz = cv::countNonZero(diff_mask);
        cv::imwrite(base + "_maskcuda.png", out_mask_cuda);
        cv::imwrite(base + "_maskdiff.png", diff_mask);
        printf("close(masked): cuda=%.3f ms (event), wall=%.3f ms, max_abs_diff=%.0f, nonzero=%d\n", cuda_ms_mask,
               wall_ms_mask, maxv, nz);
    }

    // 5) 顶帽：原图 - 开
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        ops::morphTopHat<uint8_t>(d_in, d_tmp, d_tmp2, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        auto t1 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, ev_start, ev_stop));
        wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    {
        cv::Mat out_cuda(img_h, img_w, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(out_cuda.data, d_out, bytes, cudaMemcpyDeviceToHost));

        cv::Mat out_cv;
        auto    t0_cv = std::chrono::high_resolution_clock::now();
        cv::morphologyEx(img, out_cv, cv::MORPH_TOPHAT, se, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        auto t1_cv = std::chrono::high_resolution_clock::now();
        cv_ms      = std::chrono::duration<double, std::milli>(t1_cv - t0_cv).count();

        cv::Mat diff;
        cv::absdiff(out_cuda, out_cv, diff);
        double minv = 0.0, maxv = 0.0;
        cv::minMaxLoc(diff, &minv, &maxv);
        int nz = cv::countNonZero(diff);

        std::string base = std::string(output_dir) + "/tophat";
        cv::imwrite(base + "_cuda.png", out_cuda);
        cv::imwrite(base + "_cv.png", out_cv);
        cv::imwrite(base + "_diff.png", diff);
        printf("tophat: cuda=%.3f ms (event), wall=%.3f ms, opencv=%.3f ms, max_abs_diff=%.0f, nonzero=%d\n", cuda_ms,
               wall_ms, cv_ms, maxv, nz);

        // 带掩码 CUDA 对比
        float  cuda_ms_mask = 0.0f;
        double wall_ms_mask = 0.0;
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventRecord(ev_start, stream));
            ops::morphTopHat_u8_masked(d_in, d_tmp, d_tmp2, d_out, img_w, img_h, img_stride, d_se, se_w, se_h, anchor_x,
                                       anchor_y, block_dim, stream);
            CUDA_CHECK(cudaEventRecord(ev_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            auto t1 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventElapsedTime(&cuda_ms_mask, ev_start, ev_stop));
            wall_ms_mask = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        cv::Mat out_mask_cuda(img_h, img_w, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(out_mask_cuda.data, d_out, bytes, cudaMemcpyDeviceToHost));

        cv::Mat diff_mask;
        cv::absdiff(out_mask_cuda, out_cv, diff_mask);
        cv::minMaxLoc(diff_mask, &minv, &maxv);
        nz = cv::countNonZero(diff_mask);
        cv::imwrite(base + "_maskcuda.png", out_mask_cuda);
        cv::imwrite(base + "_maskdiff.png", diff_mask);
        printf("tophat(masked): cuda=%.3f ms (event), wall=%.3f ms, max_abs_diff=%.0f, nonzero=%d\n", cuda_ms_mask,
               wall_ms_mask, maxv, nz);
    }

    // 6) 黑帽：闭 - 原图
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        ops::morphBlackHat<uint8_t>(d_in, d_tmp, d_tmp2, d_out, img_w, img_h, img_stride, radius, block_dim, stream);
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        auto t1 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, ev_start, ev_stop));
        wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    {
        cv::Mat out_cuda(img_h, img_w, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(out_cuda.data, d_out, bytes, cudaMemcpyDeviceToHost));

        cv::Mat out_cv;
        auto    t0_cv = std::chrono::high_resolution_clock::now();
        cv::morphologyEx(img, out_cv, cv::MORPH_BLACKHAT, se, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        auto t1_cv = std::chrono::high_resolution_clock::now();
        cv_ms      = std::chrono::duration<double, std::milli>(t1_cv - t0_cv).count();

        cv::Mat diff;
        cv::absdiff(out_cuda, out_cv, diff);
        double minv = 0.0, maxv = 0.0;
        cv::minMaxLoc(diff, &minv, &maxv);
        int nz = cv::countNonZero(diff);

        std::string base = std::string(output_dir) + "/blackhat";
        cv::imwrite(base + "_cuda.png", out_cuda);
        cv::imwrite(base + "_cv.png", out_cv);
        cv::imwrite(base + "_diff.png", diff);
        printf("blackhat: cuda=%.3f ms (event), wall=%.3f ms, opencv=%.3f ms, max_abs_diff=%.0f, nonzero=%d\n", cuda_ms,
               wall_ms, cv_ms, maxv, nz);

        // 带掩码 CUDA 对比
        float  cuda_ms_mask = 0.0f;
        double wall_ms_mask = 0.0;
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventRecord(ev_start, stream));
            ops::morphBlackHat_u8_masked(d_in, d_tmp, d_tmp2, d_out, img_w, img_h, img_stride, d_se, se_w, se_h,
                                         anchor_x, anchor_y, block_dim, stream);
            CUDA_CHECK(cudaEventRecord(ev_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            auto t1 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventElapsedTime(&cuda_ms_mask, ev_start, ev_stop));
            wall_ms_mask = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        cv::Mat out_mask_cuda(img_h, img_w, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(out_mask_cuda.data, d_out, bytes, cudaMemcpyDeviceToHost));

        cv::Mat diff_mask;
        cv::absdiff(out_mask_cuda, out_cv, diff_mask);
        cv::minMaxLoc(diff_mask, &minv, &maxv);
        nz = cv::countNonZero(diff_mask);
        cv::imwrite(base + "_maskcuda.png", out_mask_cuda);
        cv::imwrite(base + "_maskdiff.png", diff_mask);
        printf("blackhat(masked): cuda=%.3f ms (event), wall=%.3f ms, max_abs_diff=%.0f, nonzero=%d\n", cuda_ms_mask,
               wall_ms_mask, maxv, nz);
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
