#include "vector_add.h"

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAddKernel(float *A, float *B, float *C, int nb_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nb_elements)
    {
        // printf("i: %d, A: %f, B: %f\n", i, A[i], B[i]);
        C[i] = A[i] + B[i];
    }
}

void vectorAdd(float *A, float *B, float *C, int nb_elements)
{
    int threads_per_block = 256;
    int blocks_per_grid   = (nb_elements + threads_per_block - 1) / threads_per_block;
    vectorAddKernel<<<blocks_per_grid, threads_per_block>>>(A, B, C, nb_elements);
}
