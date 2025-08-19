#pragma once

#include <cuda_runtime.h>

#include <iostream>

// Use the CUDA runtime API to check for errors during kernel launches.
#define CUDA_CHECK(call)                                                                     \
    do                                                                                       \
    {                                                                                        \
        cudaError_t status = call;                                                           \
        if (status != cudaSuccess)                                                           \
        {                                                                                    \
            std::cerr << "CUDA error in file '" << __FILE__ << "' line " << __LINE__ << ": " \
                      << cudaGetErrorString(status) << std::endl;                            \
            std::exit(EXIT_FAILURE);                                                         \
        }                                                                                    \
    }                                                                                        \
    while (0)