#include "common/logging.h"
#include "common/plugin.h"
#include "group_normalization_plugin/group_normalization_plugin.h"
#include "utility.h"

#include <gtest/gtest.h>

#include <iostream>

#define LOG_LEVEL ILogger::Severity::kINFO

#define INPUT_NODE  "input"
#define OUTPUT_NODE "output"

using namespace nvinfer1;

UniquePtr<ICudaEngine> createNetworkWithGroupNormalization(RunTimeLogger &logger, int num_groups, int num_channels,
                                                           Dims4 input_dims)
{
    int batchSize = input_dims.d[0];

    // Create a TensorRT engine
    auto builder = makeUnique<IBuilder>(createInferBuilder(logger));
    EXPECT_NE(builder, nullptr);

    // Create network
    auto network = makeUnique<INetworkDefinition>(builder->createNetworkV2(1U));
    EXPECT_NE(network, nullptr);

    // Create input tensor
    ITensor *input = network->addInput(INPUT_NODE, DataType::kFLOAT, input_dims);
    EXPECT_NE(input, nullptr);

    // TODO: create gn Weights

    // Add plugin to the network
    auto gn = nvinfer1::plugin::addGroupNormLayer(network.get(), *input, num_groups, num_channels, 1e-5);
    EXPECT_NE(gn, nullptr);

    auto output = gn->getOutput(0);
    EXPECT_NE(gn, nullptr);

    // Set output name
    output->setName(OUTPUT_NODE);

    // Mark output tensor
    network->markOutput(*output);

    // Build engine
    builder->setMaxBatchSize(batchSize);
    builder->setMaxWorkspaceSize(1 << 30);
    auto engine = makeUnique<ICudaEngine>(builder->buildCudaEngine(*network));
    EXPECT_NE(engine, nullptr);

    // Destroy network and builder
    // using unique_ptr, no need destory manually

    return engine;
}

// clang-format off

// clang-format on

class Instance2 : public testing::TestWithParam<int>
{
};

TEST_P(Instance2, run)
{
    int n;
    n = GetParam();
    EXPECT_TRUE(n);
}

INSTANTIATE_TEST_CASE_P(GroupNormalizationPlugin, Instance2, testing::Values(1, 1, 22, 2));

// TEST(GroupNormalizationPlugin, build)
// {
//     constexpr int batch   = 1;
//     constexpr int channel = 4;
//     constexpr int height  = 2;
//     constexpr int width   = 2;
//     constexpr int volume  = batch * channel * height * width;

//     int num_groups   = 2;
//     int num_channels = channel;

//     // create a logger
//     auto logger = RunTimeLogger("GN build", LOG_LEVEL);

//     auto engine
//         = createNetworkWithGroupNormalization(logger, num_groups, num_channels, Dims4{batch, channel, height, width});
//     EXPECT_NE(engine, nullptr);

//     // Create execution context
//     auto context = makeUnique<IExecutionContext>(engine->createExecutionContext());
//     EXPECT_NE(context, nullptr);
// }

// class FooTest : public testing::TestWithParam<float>
// {
// };

// TEST_P(GroupNormalizationPlugin, run)
// {
//     constexpr int batch   = 1;
//     constexpr int channel = 4;
//     constexpr int height  = 2;
//     constexpr int width   = 2;
//     constexpr int volume  = batch * channel * height * width;

//     int num_groups   = 2;
//     int num_channels = channel;

//     // create a logger
//     auto logger = RunTimeLogger("GN build", LOG_LEVEL);

//     auto engine
//         = createNetworkWithGroupNormalization(logger, num_groups, num_channels, Dims4{batch, channel, height, width});
//     EXPECT_NE(engine, nullptr);

//     // Create execution context
//     auto context = makeUnique<IExecutionContext>(engine->createExecutionContext());
//     EXPECT_NE(context, nullptr);

//     // Create input buffer
//     // float inputData[volume]
//     //     = {1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0., 2., 2., 0., 0., 2., 0., 1., 1.,
//     //        1., 0., 0., 2., 2., 1., 1., 0., 3., 1., 2., 2., 3., 0., 0., 2., 2., 3., 1., 2., 3., 3., 2., 1.};

//     float inputData[volume] = {1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0.};
//     float expectedResult[volume]
//         = {0.57733488,  0.57733488,  0.57733488,  0.57733488, -1.73200464, 0.57733488, 0.57733488,  -1.73200464,
//            -0.77458012, -0.77458012, -0.77458012, 1.29096687, 1.29096687,  1.29096687, -0.77458012, -0.77458012};

//     // Create output buffer with the same size as inputData

//     size_t inputSize = sizeof(inputData);

//     auto input_buf  = std::make_shared<CudaBuffer<float>>(volume);
//     auto output_buf = std::make_shared<CudaBuffer<float>>(volume);

//     void *buffers[2] = {input_buf->mPtr, output_buf->mPtr};

//     float outputData[volume];

//     // Create CUDA stream
//     cudaStream_t stream;
//     EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

//     // copy input buffer to device
//     EXPECT_EQ(cudaSuccess, cudaMemcpyAsync(buffers[0], inputData, inputSize, cudaMemcpyHostToDevice, stream));

//     // Execute inference
//     context->enqueue(batch, buffers, stream, nullptr);

//     // copy output to buffer
//     EXPECT_EQ(cudaSuccess, cudaMemcpyAsync(outputData, buffers[1], inputSize, cudaMemcpyDeviceToHost, stream));

//     // Synchronize inference
//     EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

//     EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

//     // // Print the output values
//     // for (int i = 0; i < volume; ++i)
//     // {
//     //     std::cout << outputData[i] << " ";
//     // }
//     // std::cout << std::endl;

//     EXPECT_TRUE(buffersEqual(outputData, expectedResult, volume, 1e-5));
// }
