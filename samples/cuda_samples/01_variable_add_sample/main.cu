#include "CrashHandler.h"

#include <cuda_runtime.h>
#include <stdio.h>

#include <exception>

#define CHECK_CUDA(call)                                                                                           \
    {                                                                                                              \
        const cudaError_t error = call;                                                                            \
        if (error != cudaSuccess)                                                                                  \
        {                                                                                                          \
            printf("get cuda error at func: %s, line: %d,\nerror code:%d, error msg:%s\n", __FUNCTION__, __LINE__, \
                   error, cudaGetErrorString(error));                                                              \
            exit(-1);                                                                                              \
        }                                                                                                          \
    }

__global__ void gpu_add_kernel(int a, int b, int *c)
{
    *c = a + b;
}

__global__ void gpu_add_kernel_reference(int *a, int *b, int *c)
{
    *c = *a + *b;
}

void cuda_sample_pass_by_value()
{
    int  a = 10;
    int  b = 20;
    int  h_c;
    int *d_c{nullptr};
    // 为c分配设备内存 (字节)
    // 注意: 主机端不能对cudaMalloc分配的内存进行操作
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(int)));
    // 执行gpu加法, 以值传递方式
    gpu_add_kernel<<<1, 1>>>(a, b, d_c);
    // 拷贝结果到主机端, 此处会同步等待设备端计算完成
    CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost));
    printf("result of kernel pass by value, %d + %d = %d\n", a, b, h_c);
    // 释放设备端内存
    CHECK_CUDA(cudaFree(d_c));
    d_c = nullptr;
}

void cuda_sample_pass_by_reference()
{
    int  h_a = 10;
    int  h_b = 20;
    int  h_c;
    int *d_a{nullptr};
    int *d_b{nullptr};
    int *d_c{nullptr};
    // 为a, b, c分配设备内存
    CHECK_CUDA(cudaMalloc(&d_a, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_b, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(int)));
    // 拷贝数据到设备端
    CHECK_CUDA(cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice));
    // 执行gpu加法
    gpu_add_kernel_reference<<<1, 1>>>(d_a, d_b, d_c);
    // 拷贝结果到主机端
    CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost));
    printf("result of kernel pass by reference, %d + %d = %d\n", h_a, h_b, h_c);
    // 释放设备端内存
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    d_a = nullptr;
    d_b = nullptr;
    d_c = nullptr;
}

void cuda_error_sample()
{
    try
    {
        int  h_a = 10;
        int *d_a{nullptr};
        cudaMalloc(&d_a, sizeof(int));
        cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
        printf("h_a = %d\n", h_a);
        // 错误用法, 主机直接访问设备端内存
        printf("d_a = %d\n", *d_a);
        CHECK_CUDA(cudaFree(d_a));
        d_a = nullptr;
    }
    catch (const std::exception &e)
    {
        printf("catch exception: %s\n", e.what());
    }
    catch (...)
    {
        printf("catch unknown exception\n");
    }
    CHECK_CUDA(cudaGetLastError());
}

void get_cuda_error_callback()
{
    printf("get_cuda_error_callback called\n");
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("get cuda error at func: %s, line: %d,\nerror code:%d, error msg:%s\n", __FUNCTION__, __LINE__, error,
               cudaGetErrorString(error));
    }
}

int main()
{
    free_kick::common::CrashHandler crash_handler;
    crash_handler.setup(get_cuda_error_callback);
    cuda_sample_pass_by_value();
    cuda_sample_pass_by_reference();
    cuda_error_sample();
    return 0;
}