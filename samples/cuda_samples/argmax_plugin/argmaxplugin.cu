#include "argmaxplugin.h"
#include "cuda_runtime.h"

__global__ void argmaxKernel(const float *input, float *outputMaxValue, float *outputIndex, const int axisSize,
                             const int stride, const int numOutElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numOutElements)
        return;
    float maxValue = input[idx];
    int   tmpIndex = idx;
    int   maxIndex = 0;
    for (int i = 1; i < axisSize; i++)
    {
        tmpIndex += stride;
        if (input[tmpIndex] > maxValue)
        {
            maxValue = input[tmpIndex];
            maxIndex = i;
        }
    }
    // printf("idx: %d, maxValue: %f, maxIndex: %d\n", idx, maxValue, maxIndex);
    outputMaxValue[idx] = maxValue;
    outputIndex[idx]    = maxIndex;
}

namespace nvinfer1 {

ArgMaxPlugin::ArgMaxPlugin(const int dim, const bool keepdim)
    : dim(dim)
    , keepdim(keepdim)
{
}

ArgMaxPlugin::ArgMaxPlugin(const void *data, const size_t length)
{
    // 从序列化数据中加载参数
    const char *d = static_cast<const char *>(data);

    dim = *reinterpret_cast<const int *>(d);
    d += sizeof(int);
    keepdim = *reinterpret_cast<const bool *>(d);
}

DimsExprs ArgMaxPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs,
                                            IExprBuilder &exprBuilder) noexcept
{
    // 创建输出维度表达式
    DimsExprs output;
    output.nbDims = keepdim ? inputs[0].nbDims : inputs[0].nbDims - 1;

    // 根据输出索引返回不同的维度
    if (outputIndex == 0 || outputIndex == 1)
    {
        // 复制输入维度
        for (int i = 0; i < inputs[0].nbDims; i++)
        {
            if (i == dim && !keepdim)
            {
                // 如果不保持维度，则跳过该维度
                continue;
            }
            if (i == dim && keepdim)
            {
                // 如果保持维度，则该维度大小为1
                output.d[i] = exprBuilder.constant(1);
            }
            else
            {
                // 其他维度保持不变
                output.d[i] = inputs[0].d[i];
            }
        }
    }

    return output;
}

int32_t ArgMaxPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
                              const void *const *inputs, void *const *outputs, void *workspace,
                              cudaStream_t stream) noexcept
{
    // 获取输入和输出指针
    const float *input          = static_cast<const float *>(inputs[0]);
    float       *outputMaxValue = static_cast<float *>(outputs[0]);
    float       *outputIndex    = static_cast<float *>(outputs[1]);

    // 计算维度信息
    int numElements    = 1;
    int numOutElements = 1;
    int axisSize       = 1;
    int stride         = 1;

    // 计算元素数和步长
    printf("enqueue inputDims: ");
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
    {
        printf("%d ", inputDesc[0].dims.d[i]);
        if (i == dim)
        {
            axisSize = inputDesc[0].dims.d[i]; // 2
        }
        else
        {
            stride *= inputDesc[0].dims.d[i];
        }
        numElements *= inputDesc[0].dims.d[i];
    }
    printf("\n");
    printf("enqueue outputDims: ");
    for (int i = 0; i < outputDesc[0].dims.nbDims; i++)
    {
        printf("%d ", outputDesc[0].dims.d[i]);
    }
    printf("\n");
    numOutElements = numElements / axisSize;

    // 配置 CUDA kernel 启动参数
    const int blockSize = 256;
    const int gridSize  = (numOutElements + blockSize - 1) / blockSize;
    printf("gridSize: %d, blockSize: %d, numOutElements: %d, axisSize: %d, stride: %d, numElements: %d\n", gridSize,
           blockSize, numOutElements, axisSize, stride, numElements);

    // 启动 kernel
    argmaxKernel<<<gridSize, blockSize, 0, stream>>>(input, outputMaxValue, outputIndex, axisSize, stride,
                                                     numOutElements);

    return 0;
}

void ArgMaxPlugin::serialize(void *buffer) const noexcept
{
    char *d = static_cast<char *>(buffer);

    *reinterpret_cast<int *>(d) = dim;
    d += sizeof(int);
    *reinterpret_cast<bool *>(d) = keepdim;
}

DataType ArgMaxPlugin::getOutputDataType(int32_t index, const DataType *inputTypes, int32_t nbInputs) const noexcept
{
    return DataType::kFLOAT;
}

size_t ArgMaxPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) + sizeof(bool);
}

IPluginV2 *ArgMaxPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    int  dim     = 0;
    bool keepdim = false;

    // 从 PluginFieldCollection 中解析参数
    for (int i = 0; i < fc->nbFields; i++)
    {
        const char *name = fc->fields[i].name;
        const void *data = fc->fields[i].data;

        if (!strcmp(name, "dim"))
        {
            dim = *(static_cast<const int *>(data));
        }
        else if (!strcmp(name, "keepdim"))
        {
            keepdim = *(static_cast<const bool *>(data));
        }
    }

    return new ArgMaxPlugin(dim, keepdim);
}

IPluginV2 *ArgMaxPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                                  size_t serialLength) noexcept
{
    return new ArgMaxPlugin(serialData, serialLength);
}

IPluginV2Layer *addArgMaxLayer(INetworkDefinition *network, ITensor *input, int dim, bool keepdim)
{
    // Create plugin field collection
    static PluginField fields[2];

    fields[0].name   = "dim";
    fields[0].type   = PluginFieldType::kINT32;
    fields[0].data   = &dim;
    fields[0].length = 1;

    fields[1].name   = "keepdim";
    fields[1].type   = PluginFieldType::kINT8;
    fields[1].data   = &keepdim;
    fields[1].length = 1;

    static PluginFieldCollection fc{2, fields};

    // Create plugin
    static ArgMaxPluginCreator creator;

    auto plugin = creator.createPlugin("argmax", &fc);
    // Add plugin layer to network
    return network->addPluginV2(&input, 1, *plugin);
}
} // namespace nvinfer1