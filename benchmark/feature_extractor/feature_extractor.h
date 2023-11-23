#pragma once
#include <NvInfer.h>

#include <string>

class FeatureExtractor
{
public:
    FeatureExtractor(const std::string &backbone, const std::string &engine_path);
    ~FeatureExtractor();

private:
    int initModel(const std::string &engine_path);
    int initData();

    int                          device_{0};
    nvinfer1::IRuntime          *runtime_{nullptr};
    nvinfer1::ICudaEngine       *engine_{nullptr};
    nvinfer1::IExecutionContext *context_{nullptr};
};
