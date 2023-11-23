#include "feature_extractor.h"

#include "helper.h"

#include <cuda_runtime.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

static Logger gLogger(nvinfer1::ILogger::Severity::kWARNING);

FeatureExtractor::FeatureExtractor(const std::string &backbone, const std::string &engine_path)
{
    int error_code = initModel(engine_path);
    if (error_code != 0)
    {
        return;
    }
}

int FeatureExtractor::initModel(const std::string &engine_path)
{
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file.good())
    {
        std::cerr << __FUNCTION__ << ": Could not read engine from file." << std::endl;
        return -1;
    }
    engine_file.seekg(0, engine_file.end);
    size_t fsize = engine_file.tellg() < 0 ? 0 : engine_file.tellg();
    assert(fsize != 0 && "Could not read engine with empty file.");
    engine_file.seekg(0, engine_file.beg);
    std::vector<char> engine_string(fsize);
    engine_file.read(engine_string.data(), fsize);
    engine_file.close();

    cudaSetDevice(device_);

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    assert(runtime_ != nullptr && "Failed to create infer runtime.");
    engine_ = runtime_->deserializeCudaEngine(engine_string.data(), fsize);
    assert(engine_ != nullptr && "Failed to deserialize cuda engine.");
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr && "Failed to create execution context.");

    return 0;
}

int FeatureExtractor::initData()
{
    engine_->getbin
}
