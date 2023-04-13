#include "common/logging.h"
#include "group_normalization_plugin/group_normalization_plugin.h"
#include <gtest/gtest.h>
#include <iostream>

// Use the CUDA runtime API to check for errors during kernel launches.
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            std::cerr << "CUDA error in file '" << __FILE__ << "' line " << __LINE__ << ": "                           \
                      << cudaGetErrorString(status) << std::endl;                                                      \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (0)

// Execute inference

// Compare two buffers for equality within a specified tolerance
bool buffersEqual(const void *buf1, const void *buf2, size_t size, float tolerance) {
    const float *f1 = static_cast<const float *>(buf1);
    const float *f2 = static_cast<const float *>(buf2);
    for (size_t i = 0; i < size / sizeof(float); ++i) {
        if (std::abs(f1[i] - f2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

namespace nvinfer1 {

TEST(GroupNormalizationPlugin, build) {
    constexpr int batch = 3;
    constexpr int channel = 4;
    constexpr int height = 2;
    constexpr int width = 2;
    constexpr int volume = batch * channel * height * width;

    auto logger = RunTimeLogger();
    std::cout << "Create a TensorRT engine" << std::endl;
    // Create a TensorRT engine
    IBuilder *builder = createInferBuilder(logger);

    std::cout << "Create network" << std::endl;
    INetworkDefinition *network = builder->createNetworkV2(1U);

    std::cout << "Create input tensor" << std::endl;
    // Create input tensor
    ITensor *input = network->addInput("input", DataType::kFLOAT, Dims4{batch, channel, height, width});

    std::cout << "Add plugin to the network" << std::endl;
    auto gn = nvinfer1::plugin::addGroupNormLayer(network, *input, 3, 1e-5);
    std::cout << "GN plugin: " << (void *)gn << std::endl;
    std::cout << "getOutput" << std::endl;
    auto output = gn->getOutput(0);

    std::cout << "Set output name" << std::endl;
    output->setName("output");

    std::cout << "Mark output tensor" << std::endl;
    network->markOutput(*output);

    std::cout << "Build engine" << std::endl;
    builder->setMaxBatchSize(batch);
    builder->setMaxWorkspaceSize(1 << 30);
    ICudaEngine *engine = builder->buildCudaEngine(*network);

    std::cout << "Destroy network and builder" << std::endl;
    network->destroy();
    builder->destroy();

    std::cout << "Create execution context" << std::endl;
    IExecutionContext *context = engine->createExecutionContext();

    std::cout << "Create input buffer" << std::endl;
    float inputData[volume] = {1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0.,
                               2., 2., 0., 0., 2., 0., 1., 1., 1., 0., 0., 2., 2., 1., 1., 0.,
                               3., 1., 2., 2., 3., 0., 0., 2., 2., 3., 1., 2., 3., 3., 2., 1.};

    // Create output buffer with the same size as inputData
    float outputData[volume];

    void *buffers[2];
    size_t inputSize = sizeof(inputData);
    CUDA_CHECK(cudaMalloc(&buffers[0], inputSize));
    CUDA_CHECK(cudaMallocHost(&buffers[1], inputSize));

    std::cout << "Create CUDA stream" << std::endl;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "copy input buffer to device" << std::endl;
    cudaMemcpyAsync(buffers[0], inputData, inputSize, cudaMemcpyHostToDevice, stream);

    std::cout << "Execute inference" << std::endl;
    context->enqueue(batch, buffers, stream, nullptr);

    std::cout << "copy output to buffer" << std::endl;
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], inputSize, cudaMemcpyDeviceToHost, stream));

    std::cout << "Synchronize inference" << std::endl;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Print the output values
    for (int i = 0; i < volume; ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Destroy execution context and engine" << std::endl;
    context->destroy();
    engine->destroy();
}

} // namespace nvinfer1
