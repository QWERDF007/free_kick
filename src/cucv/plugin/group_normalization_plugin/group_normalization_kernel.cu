

#include "group_normalization_plugin.h"
#include "stdio.h"

namespace nvinfer1 { namespace plugin {

/**
 * 这段代码是一个用于 Group Normalization 的 Scale-Shift 操作的 CUDA 核函数实现。
 * 它接受一个输入张量 inOut，以及张量的维度 ld、B 和 C。它还接受 beta 和 gamma 参数，用于缩放和移位输入张量。
 * 该核函数使用大小为 (colBlocks, C, B) 的网格启动，其中 colBlocks 是每列的块数，C 是通道数，B 是批次大小。
 * 然后，核函数对输入张量的每个元素执行 Scale-Shift 操作，使用每个通道对应的 beta 和 gamma 值。
 */
template<typename T, unsigned TPB>
__global__ void scaleShiftChannelsInplaceKernel(T *inOut, const int ld, float *beta, float *gamma)
{
    // grid is blocks x C x B
    // ld should be H*W
    // blockIdx.z = batch
    // blockIdx.y = channel
    // blockIdx.x = block per col
    const T b = beta[blockIdx.y];
    const T g = gamma[blockIdx.y];

    const int offset = (blockIdx.z * gridDim.y + blockIdx.y) * ld;

    const int tx = blockIdx.x * TPB + threadIdx.x;
    if (tx < ld)
    {
        inOut[offset + tx] = g * inOut[offset + tx] + b;
    }
}

/**
 * scaleShiftChannelsInplace 函数是一个包装函数，它首先计算每列的块数，然后使用适当的网格和块大小启动核函数。
 * 它接受一个输入张量 inOut，以及张量的维度 ld、B 和 C。它还接受 beta 和 gamma 参数，用于缩放和移位输入张量。
 * 以及一个 CUDA 流 stream，启动用于 Group Normalization 的 Scale-Shift (尺度变换) 操作的 CUDA 核函数。
 */
template<typename T>
cudaError_t scaleShiftChannelsInplace(T *inOut, const int B, const int C, const int channelVolume, float *beta,
                                      float *gamma, cudaStream_t stream)
{
    // TPB 线程块大小
    constexpr int TPB       = 256;
    const int     colBlocks = (channelVolume + TPB - 1) / TPB;
    const dim3    grid(colBlocks, C, B);
    scaleShiftChannelsInplaceKernel<T, TPB><<<grid, TPB, 0, stream>>>(inOut, channelVolume, beta, gamma);
    return cudaPeekAtLastError();
}

template cudaError_t scaleShiftChannelsInplace<float>(float *inOut, const int B, const int C, const int channelVolume,
                                                      float *beta, float *gamma, cudaStream_t stream);
}} // namespace nvinfer1::plugin
