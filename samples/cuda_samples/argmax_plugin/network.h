#pragma once

#include <vector>
#include <memory>
#include <string>
#include <NvInfer.h>
#include <cuda_runtime.h>

class Network
{
public:
    Network();
    ~Network();
    void build();
    void run();

private:
    void buildModel();
    void buildData();

    struct InferDeleter
    {
        template<typename T>
        void operator()(T *obj) const
        {
            delete obj;
        }
    };

    std::shared_ptr<nvinfer1::IRuntime>          runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine>       engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    void* input_device_{nullptr};
    void* output_device_{nullptr};
    void* index_device_{nullptr};

    int input_size_{0};
    int output_size_{0};
    int index_size_{0};

    std::vector<float> input_data_;
    std::vector<float> output_host_;
    std::vector<float> index_host_;
    std::vector<void*> bindings_;

    cudaStream_t stream_{nullptr};

    inline static std::string input_name{"INPUT"};
    inline static std::string output_name{"OUTPUT"};
    inline static std::string output2_name{"OUTPUT2"};
    inline static int input_width = 4;
    inline static int input_height = 4;
    inline static int input_ch = 2;
};