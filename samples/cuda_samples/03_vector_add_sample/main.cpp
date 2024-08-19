#include "vector_add.h"

#include <cuda_runtime.h>
#include <stdio.h>

#include <chrono>
#include <memory>

#define CUDA_CHECK_RETURN(ret)                                                                                \
    if (ret != cudaSuccess)                                                                                   \
    {                                                                                                         \
        fprintf(stderr, "CUDA error: %s (%s at line %d)\n", cudaGetErrorString(ret), __FUNCTION__, __LINE__); \
        exit(EXIT_FAILURE);                                                                                   \
    }

int main(int argc, char *argv[])
{
    cudaError_t err = cudaSuccess;

    // 设置矩阵大小和计算字节大小
    int    nb_elements = 500000001;
    size_t nb_bytes    = nb_elements * sizeof(float);
    printf("[Vector addition of %d elements]\n", nb_elements);

    // 为矩阵A/B/C分配主机内存
    // std::vector<float> h_A(num_elements);
    float *h_A       = (float *)malloc(nb_bytes);
    float *h_B       = (float *)malloc(nb_bytes);
    float *cpu_out_C = (float *)malloc(nb_bytes);
    float *gpu_out_C = (float *)malloc(nb_bytes);

    // 初始化矩阵A/B的值
    for (int i = 0; i < nb_elements; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 计算CPU结果
    printf("Calculate the CPU result\n");
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nb_elements; i++)
    {
        cpu_out_C[i] = h_A[i] + h_B[i];
    }
    auto   end_time        = std::chrono::high_resolution_clock::now();
    double wall_clock_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    printf("CPU calculation time: %f ms\n", wall_clock_time);

    // 为矩阵A/B/C分配设备内存
    float *d_A = nullptr;
    err        = cudaMalloc((void **)&d_A, nb_bytes);
    CUDA_CHECK_RETURN(err);

    float *d_B = NULL;
    err        = cudaMalloc((void **)&d_B, nb_bytes);
    CUDA_CHECK_RETURN(err);

    float *d_C = NULL;
    err        = cudaMalloc((void **)&d_C, nb_bytes);
    CUDA_CHECK_RETURN(err);

    // 从主机内存拷贝A/B/C到设备内存
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, nb_bytes, cudaMemcpyHostToDevice);
    CUDA_CHECK_RETURN(err);

    err = cudaMemcpy(d_B, h_B, nb_bytes, cudaMemcpyHostToDevice);
    CUDA_CHECK_RETURN(err);

    // 运行 vectorAdd CUDA 核函数
    int threads_per_block = 256;
    int blocks_per_grid   = (nb_elements + threads_per_block - 1) / threads_per_block;
    printf("Launching CUDA kernel with %d blocks of %d threads\n", blocks_per_grid, threads_per_block);
    start_time = std::chrono::high_resolution_clock::now();
    vectorAdd(d_A, d_B, d_C, nb_elements);
    err = cudaGetLastError();
    CUDA_CHECK_RETURN(err);
    // 同步结果
    // err             = cudaDeviceSynchronize();
    end_time        = std::chrono::high_resolution_clock::now();
    wall_clock_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    printf("CUDA kernel time: %f ms\n", wall_clock_time);
    CUDA_CHECK_RETURN(err);

    // CUDA_CHECK_RETURN(err);

    // 从设备内存拷贝C到主机内存
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(gpu_out_C, d_C, nb_bytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK_RETURN(err);

    // 验证结果是否正确
    for (int i = 0; i < nb_elements; i++)
    {
        if (fabs(cpu_out_C[i] - gpu_out_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d, cpu: %f, gpu: %f!\n", i, cpu_out_C[i],
                    gpu_out_C[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    printf("Freeing device resources\n");
    // 释放设备内存
    err = cudaFree(d_A);
    CUDA_CHECK_RETURN(err);
    err = cudaFree(d_B);
    CUDA_CHECK_RETURN(err);
    err = cudaFree(d_C);
    CUDA_CHECK_RETURN(err);

    printf("Freeing host resources\n");
    // 释放主机内存
    free(h_A);
    free(h_B);
    free(cpu_out_C);
    free(gpu_out_C);
    printf("Done!\n");
    return 0;
}