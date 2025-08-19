#include "common/utility.h"

#include <cuda_runtime.h>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <stdio.h>

#include <chrono>

struct CuComplex
{
    float r;
    float i;

    __host__ __device__ CuComplex(float a, float b)
        : r(a)
        , i(b)
    {
    }

    __host__ __device__ float magnitude2()
    {
        return r * r + i * i;
    }

    __host__ __device__ CuComplex operator*(const CuComplex &other)
    {
        return CuComplex(r * other.r - i * other.i, i * other.r + r * other.i);
    }

    __host__ __device__ CuComplex operator+(const CuComplex &other)
    {
        return CuComplex(r + other.r, i + other.i);
    }
};

__host__ __device__ int julia(int x, int y, const int DIM)
{
    // 图像缩放因子
    const float scale = 1.5;

    // 将坐标转换为复数空间的坐标，并将原点定位到图像中心, 图像范围缩放到 [-1, 1]
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    // 定义复数常量, 此常量能够生成一个比较精美的图案
    CuComplex c(-0.8, 0.156);
    CuComplex a(jx, jy);

    // 迭代计算
    for (int i = 0; i < 200; ++i)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

void cpu_kernel(unsigned char *ptr, const int DIM, const int ch)
{
    for (int y = 0; y < DIM; y++)
    {
        for (int x = 0; x < DIM; x++)
        {
            int i = y * DIM + x;

            int julia_value = 255 * julia(x, y, DIM);

            ptr[i * ch + 0] = 0;
            ptr[i * ch + 1] = julia_value;
            ptr[i * ch + 2] = julia_value;
        }
    }
}

__global__ void gpu_kernel(unsigned char *ptr, const int DIM, const int ch)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    // printf("x = %d, y = %d, gridDim.x = %d\n", x, y, gridDim.x);
    int i = x + y * gridDim.x;

    int julia_value = 255 * julia(x, y, DIM);

    ptr[i * ch + 0] = 0;
    ptr[i * ch + 1] = julia_value;
    ptr[i * ch + 2] = julia_value;
}

void cpu_sample(const int DIM)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // 创建图像
    cv::Mat img(DIM, DIM, CV_8UC3, cv::Scalar::all(0));

    // 获取图像数据指针
    unsigned char *ptr = (unsigned char *)img.data;

    // 调用CPU内核
    cpu_kernel(ptr, DIM, 3);

    auto end_time = std::chrono::high_resolution_clock::now();
    printf("CPU time: %f ms\n", std::chrono::duration<double, std::milli>(end_time - start_time).count());

    // 显示图像
    cv::imshow("CPU Julia Set", img);
    cv::waitKey(0);
}

void gpu_sample(const int DIM)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // 创建图像

    cv::Mat img(DIM, DIM, CV_8UC3, cv::Scalar::all(0));

    size_t nb_bytes = img.total() * img.elemSize();

    unsigned char *d_img{nullptr};
    CUDA_CHECK(cudaMalloc((void **)&d_img, nb_bytes));

    // 将图像数据拷贝到设备端
    CUDA_CHECK(cudaMemcpy(d_img, img.data, nb_bytes, cudaMemcpyHostToDevice));
    // 调用GPU内核
    dim3 grid(DIM, DIM);
    gpu_kernel<<<grid, 1>>>(d_img, DIM, 3);
    // 将图像数据拷贝回主机端
    CUDA_CHECK(cudaMemcpy(img.data, d_img, nb_bytes, cudaMemcpyDeviceToHost));

    // 释放设备端内存
    CUDA_CHECK(cudaFree(d_img));

    auto end_time = std::chrono::high_resolution_clock::now();
    printf("GPU time: %f ms\n", std::chrono::duration<double, std::milli>(end_time - start_time).count());

    // 显示图像
    cv::imshow("GPU Julia Set", img);
    cv::waitKey(0);
}

int main(int argc, char *argv[])
{
    const int DIM = 1024;
    cpu_sample(DIM);
    gpu_sample(DIM);
    return 0;
}