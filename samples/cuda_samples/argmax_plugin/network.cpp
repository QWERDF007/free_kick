#include "network.h"

#include "argmaxplugin.h"
#include "logging.h"

#include <exception>
#include <numeric>

namespace sample {
Logger gLogger{Logger::Severity::kVERBOSE};
} // namespace sample

Network::Network() {}

Network::~Network()
{
    if (input_device_)
        cudaFree(input_device_);
    if (output_device_)
        cudaFree(output_device_);
    if (index_device_)
        cudaFree(index_device_);
    if (stream_)
        cudaStreamDestroy(stream_);
}

void Network::build()
{
    std::cout << __FUNCTION__ << " " << __LINE__ << std::endl;
    buildModel();
    std::cout << __FUNCTION__ << " " << __LINE__ << std::endl;
    buildData();
    std::cout << __FUNCTION__ << " " << __LINE__ << std::endl;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec)
{
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        os << vec[i];
        if (i < vec.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

void Network::run()
{
    std::cout << __FUNCTION__ << " " << __LINE__ << std::endl;
    // Copy input data to GPU
    cudaMemcpyAsync(input_device_, input_data_.data(), input_data_.size() * sizeof(float), cudaMemcpyHostToDevice,
                    stream_);

    // Execute inference
    context_->enqueueV2(bindings_.data(), stream_, nullptr);

    // Copy output back to host
    cudaMemcpyAsync(output_host_.data(), output_device_, output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(index_host_.data(), index_device_, index_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);

    // Synchronize stream
    cudaStreamSynchronize(stream_);
    std::cout << output_host_ << std::endl;
    std::cout << index_host_ << std::endl;
}

void Network::buildModel()
{
    std::cout << __FUNCTION__ << " " << __LINE__ << std::endl;
    // Create builder and network
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));

    // Create input tensor with dynamic batch size
    auto input = network->addInput(input_name.c_str(), nvinfer1::DataType::kFLOAT,
                                   nvinfer1::Dims4{-1, input_ch, input_height, input_height});
    auto dims  = input->getDimensions();
    std::cout << "input dims: ";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        std::cout << dims.d[i] << " ";
    }
    std::cout << std::endl;

    // get top1 max val in channel
    // auto argmax = network->addTopK(*input, nvinfer1::TopKOperation::kMAX, 1, 0x2);
    auto argmax = nvinfer1::addArgMaxLayer(network.get(), input, 1, true);

    dims = argmax->getOutput(0)->getDimensions();
    std::cout << "argmax output 0 dims: ";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        std::cout << dims.d[i] << " ";
    }
    std::cout << std::endl;

    dims = argmax->getOutput(1)->getDimensions();
    std::cout << "argmax output 1 dims: ";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        std::cout << dims.d[i] << " ";
    }
    std::cout << std::endl;

    // Add max pooling layer
    auto pool = network->addPoolingNd(*argmax->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{3, 3});
    pool->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool->setPaddingNd(nvinfer1::DimsHW{1, 1});

    // Mark output
    pool->getOutput(0)->setName(output_name.c_str());
    network->markOutput(*pool->getOutput(0));
    // 15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15
    // 15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15
    argmax->getOutput(1)->setName(output2_name.c_str());
    network->markOutput(*argmax->getOutput(1));
    // argmax->getOutput(1)->setType(nvinfer1::DataType::kFLOAT);

    // Create optimization config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    // config->setMaxWorkspaceSize(1 << 20);  // 1MB

    // Build engine
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4{1, input_ch, input_height, input_width});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4{4, input_ch, input_height, input_width});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4{8, input_ch, input_height, input_width});
    config->addOptimizationProfile(profile);

    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine)
    {
        throw std::runtime_error("Failed to build TensorRT engine");
    }

    // Create runtime and deserialize engine
    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger), InferDeleter());
    engine_  = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()), InferDeleter());
    if (!engine_)
    {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }

    // Create execution context
    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(), InferDeleter());
    if (!context_)
    {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }
    cudaStreamCreate(&stream_);
}

void Network::buildData()
{
    std::cout << __FUNCTION__ << " " << __LINE__ << std::endl;
    // Set batch size for inference
    const int batch_size = 1;
    context_->setInputShape(input_name.c_str(), nvinfer1::Dims4{batch_size, input_ch, input_height, input_width});
    // Create input data with sequential values
    auto input_dims = context_->getTensorShape(input_name.c_str());
    input_size_     = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int>());
    std::vector<float> input_data(input_height * input_width);
    std::iota(input_data.begin(), input_data.end(), 0.0f); // Fill with 0, 1, 2, ...
    // input_data_ = input_data;
    input_data_.insert(input_data_.end(), input_data.begin(), input_data.end());
    std::reverse(input_data.begin(), input_data.end());
    input_data_.insert(input_data_.end(), input_data.begin(), input_data.end());
    std::cout << input_data_ << std::endl;

    // Allocate GPU memory for input
    if (cudaMalloc(&input_device_, input_size_ * sizeof(float)) != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate GPU memory for input");
    }

    // Get output dimensions and allocate output buffer
    auto output_dims = context_->getTensorShape(output_name.c_str());
    output_size_     = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1, std::multiplies<int>());
    std::cout << "output_size_: " << output_size_ << " output dims: " << std::endl;
    for (int i = 0; i < output_dims.nbDims; ++i)
    {
        std::cout << output_dims.d[i] << " ";
    }
    std::cout << std::endl;
    std::vector<float> output_host(output_size_);
    output_host_ = output_host;
    // Allocate GPU memory for input
    if (cudaMalloc(&output_device_, output_size_ * sizeof(float)) != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate GPU memory for output");
    }

    auto index_dims = context_->getTensorShape(output2_name.c_str());
    index_size_     = std::accumulate(index_dims.d, index_dims.d + index_dims.nbDims, 1, std::multiplies<int>());
    std::cout << "index_size_: " << output_size_ << " index dims: " << std::endl;
    for (int i = 0; i < index_dims.nbDims; ++i)
    {
        std::cout << index_dims.d[i] << " ";
    }
    std::cout << std::endl;
    std::vector<float> index_host(index_size_);
    index_host_ = index_host;
    // Allocate GPU memory for input
    if (cudaMalloc(&index_device_, index_size_ * sizeof(float)) != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate GPU memory for output");
    }

    // Setup input/output bindings
    std::vector<void *> bindings{input_device_, output_device_, index_device_};
    bindings_ = bindings;
}
