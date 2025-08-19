#pragma once

#ifdef _WIN32
#    ifdef FREE_KICK_CUDA_OPS_LIBRARY
#        define CUDA_OPS_API __declspec(dllexport)
#    else
#        define CUDA_OPS_API __declspec(dllimport)
#    endif
#else
#    define CUDA_OPS_API
#endif