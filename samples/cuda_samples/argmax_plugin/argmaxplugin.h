#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace nvinfer1 {

class ArgMaxPlugin : public IPluginV2DynamicExt
{
public:
    ArgMaxPlugin(const int dim, const bool keepdim);

    ArgMaxPlugin(const void *data, const size_t length);

    ~ArgMaxPlugin() override = default;

    const char *getPluginType() const noexcept override
    {
        return "ArgMaxPlugin";
    }

    const char *getPluginVersion() const noexcept override
    {
        return "1.0";
    }

    int getNbOutputs() const noexcept override
    {
        return 2; // 一个是最大值，一个是索引
    }

    int initialize() noexcept override
    {
        return 0; // 没有特殊初始化需求
    }

    void terminate() noexcept override
    {
        // 没有特殊结束需求
    }

    void serialize(void *buffer) const noexcept override;

    void destroy() noexcept override
    {
        delete this;
    }

    size_t getSerializationSize() const noexcept override;

    void setPluginNamespace(const char *pluginNamespace) noexcept
    {
        mNamespace = pluginNamespace;
    }

    const char *getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    /********IPluginV2Ext**********/

    DataType getOutputDataType(int32_t index, const DataType *inputTypes, int32_t nbInputs) const noexcept override;

    /********IPluginV2DynamicExt**********/

    IPluginV2DynamicExt *clone() const noexcept override
    {
        return new ArgMaxPlugin(dim, keepdim);
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs,
                                  IExprBuilder &exprBuilder) noexcept override;

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override
    {
        return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        // return true;
    }

    void configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out,
                         int32_t nbOutputs) noexcept override
    {
    }

    size_t getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs,
                            int32_t nbOutputs) const noexcept override
    {
        return 0;
    }

    int32_t enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs,
                    void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

private:
    int         dim;     // 指定计算最大值的维度
    bool        keepdim; // 是否保持维度
    std::string mNamespace;
};

// 插件创建器
class ArgMaxPluginCreator : public IPluginCreator
{
public:
    ArgMaxPluginCreator() {}

    const char *getPluginName() const noexcept override
    {
        return "ArgMaxPlugin";
    }

    const char *getPluginVersion() const noexcept override
    {
        return "1.0";
    }

    const PluginFieldCollection *getFieldNames() noexcept override
    {
        return &mFC;
    }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char *libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char *getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::string                            mNamespace;
    inline static PluginFieldCollection    mFC;
    inline static std::vector<PluginField> mPluginAttributes;
};

// 注册插件
REGISTER_TENSORRT_PLUGIN(ArgMaxPluginCreator);

// Helper function to create ArgMax plugin layer
IPluginV2Layer *addArgMaxLayer(INetworkDefinition *network, ITensor *input, int dim, bool keepdim = false);
} // namespace nvinfer1