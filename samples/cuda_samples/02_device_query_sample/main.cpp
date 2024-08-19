// #include "CrashHandler.h"

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

int main()
{
    cudaDeviceProp prop;
    int            deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    int dev;
    cudaGetDevice(&dev);
    printf("deviceCount: %d\n", deviceCount);
    for (int i = 0; i < deviceCount; i++)
    {
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        printf("Device %d\n", i);
        printf("    Name: %s\n", prop.name);
        printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("    Processor Count: %d\n", prop.multiProcessorCount);
        printf("    Clock Rate: %f\n", prop.clockRate / 1000.0);
        printf("    Memory Clock Rate: %f\n", prop.memoryClockRate / 1000.0);
        printf("    Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("    Peak Memory Bandwidth: %f GB/s\n",
               prop.memoryClockRate * 2.0 * prop.memoryBusWidth / 8.0 / 1024.0 / 1024.0);
        printf("    Total Global Memory: %f\n", prop.totalGlobalMem / 1024.0 / 1024.0);
        printf("    Total Constant Memory: %lu\n", prop.totalConstMem / 1024.0 / 1024.0);
        printf("    Shared Memory per Block: %lu KB\n", prop.sharedMemPerBlock / 1024);
        printf("    Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("    Max Thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
    }
    return 0;
}