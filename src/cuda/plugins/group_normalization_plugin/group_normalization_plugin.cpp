#include "group_normalization_plugin.h"

#include "dimsHelpers.h"
#include "serialize.hpp"

using namespace nvinfer1;
using nvinfer1::plugin::GroupNormalizationPlugin;
using nvinfer1::plugin::GroupNormalizationPluginCreator;

namespace {
constexpr const char *kGROUP_NORM_VERSION{"1"};                     // GroupNormalizationPlugin版本号
constexpr const char *kGROUP_NORM_NAME{"GroupNormalizationPlugin"}; // GroupNormalizationPlugin名称
} // namespace

// // Static class fields initialization
PluginFieldCollection              GroupNormalizationPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GroupNormalizationPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GroupNormalizationPluginCreator);

GroupNormalizationPlugin::GroupNormalizationPlugin(float epsilon, int nbGroups, int nbChannels)
    : mEpsilon(epsilon)   // 初始化mEpsilon为epsilon
    , mNbGroups(nbGroups) // 初始化mNbGroups为nbGroups
    , mNbChannels(nbChannels)
{
    PLUGIN_VALIDATE(mEpsilon > 0.0F); // 确保mEpsilon大于0
    // Number of groups should be positive
    PLUGIN_VALIDATE(mNbGroups > 0); // 确保mNbGroups大于0
    PLUGIN_VALIDATE(mNbChannels > 0);
    // TODO: 获取 gamma 和 beta 数据并 copy 到 device
}

int GroupNormalizationPlugin::initialize() noexcept
{
    auto allocScaleBias = [this](std::shared_ptr<CudaBind<float>> &buf, float value)
    {
        // 确保mNbScaleBias大于0
        PLUGIN_VALIDATE(mNbScaleBias > 0);
        if (!buf || !buf->mPtr || buf->mSize != mNbScaleBias)
        {
            // 分配设备内存
            buf = std::make_shared<CudaBind<float>>(mNbScaleBias);

            // 初始化values为长度为mNbScaleBias，值为value的向量
            std::vector<float> const values(mNbScaleBias, value);
            // 将values拷贝到buf->mPtr中
            PLUGIN_CUASSERT(cudaMemcpy(buf->mPtr, values.data(), sizeof(float) * mNbScaleBias, cudaMemcpyHostToDevice));
        }
    };

    // 分配内存并初始化mBnScales、mBnBias
    allocScaleBias(mBnScales, 1.F);
    allocScaleBias(mBnBias, 0.F);

    auto allocGammaBeta = [this](std::shared_ptr<CudaBind<float>> &buf, float value)
    {
        PLUGIN_VALIDATE(mNbChannels > 0);
        if (!buf || !buf->mPtr || buf->mSize != mNbChannels)
        {
            // 分配设备内存
            buf = std::make_shared<CudaBind<float>>(mNbChannels);

            std::vector<float> const values(mNbChannels, value);

            PLUGIN_CUASSERT(cudaMemcpy(buf->mPtr, values.data(), sizeof(float) * mNbChannels, cudaMemcpyHostToDevice));
        }
    };
    // 分配内存并初始化
    allocGammaBeta(mGnGammas, 1.F);
    allocGammaBeta(mGnBetas, 0.f);
    return 0;
}

GroupNormalizationPlugin::GroupNormalizationPlugin(const void *data, size_t length)
{
    // 反序列化，按照序列化的顺序
    deserialize_value(&data, &length, &mEpsilon);
    deserialize_value(&data, &length, &mNbGroups);
    deserialize_value(&data, &length, &mNbChannels);
    deserialize_value(&data, &length, &mNbScaleBias);
    // TODO: 反序列化 gamma 和 beta
}

const char *GroupNormalizationPlugin::getPluginType() const noexcept
{
    return kGROUP_NORM_NAME; // 返回GroupNormalizationPlugin名称
}

const char *GroupNormalizationPlugin::getPluginVersion() const noexcept
{
    return kGROUP_NORM_VERSION; // 返回GroupNormalizationPlugin版本号
}

int GroupNormalizationPlugin::getNbOutputs() const noexcept
{
    return 1; // 返回输出数量为1
}

nvinfer1::DimsExprs GroupNormalizationPlugin::getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs,
                                                                  int                     nbInputs,
                                                                  nvinfer1::IExprBuilder &exprBuilder) noexcept
{
    // 插件的三个输入分别为上一层的输入、gamma 和 beta
    // 实际推理 gamma 和 beta 应该从 weight 读取
    PLUGIN_ASSERT(nbInputs == 1);          // 确保输入数量为1
    PLUGIN_ASSERT(index == 0);             // 确保输出索引为0
    nvinfer1::DimsExprs output(inputs[0]); // 输出的维度与上一层的输入相同
    return output;                         // 返回输出的维度
}

void GroupNormalizationPlugin::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                                               IGpuAllocator *gpuAllocator) noexcept
{
    PLUGIN_ASSERT(cudnnContext);                              // 确保cudnnContext不为空
    _cudnn_handle = cudnnContext;                             // 将cudnnContext赋值给_cudnn_handle
    PLUGIN_CUDNNASSERT(cudnnCreateTensorDescriptor(&desc));   // 创建描述符desc
    PLUGIN_CUDNNASSERT(cudnnCreateTensorDescriptor(&bnDesc)); // 创建描述符bnDesc
}

// 将插件对象从其执行上下文中分离
void GroupNormalizationPlugin::detachFromContext() noexcept
{
    PLUGIN_CUDNNASSERT(cudnnDestroyTensorDescriptor(desc));   // 销毁描述符desc
    PLUGIN_CUDNNASSERT(cudnnDestroyTensorDescriptor(bnDesc)); // 销毁描述符bnDesc
}

int GroupNormalizationPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                      const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                      void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    // 获取输入维度
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    int            batchSize  = input_dims.d[0];
    int            nbChannels = input_dims.d[1];
    PLUGIN_VALIDATE(nbChannels == mNbChannels);

    // 计算每个组的大小
    int groupSize = mNbChannels / mNbGroups;

    // 计算每个通道的体积，即 height * width
    mChannelVolume = pluginInternal::volume(input_dims, /*start*/ 2, /*stop*/ inputDesc[0].dims.nbDims);

    // 设置 cudnn tensor 描述符
    PLUGIN_CHECK_CUDNN(cudnnSetTensor4dDescriptor(desc,                  // 描述符
                                                  CUDNN_TENSOR_NCHW,     // 张量格式
                                                  CUDNN_DATA_FLOAT,      // 类型
                                                  1,                     // Batchsize
                                                  batchSize * mNbGroups, // 通道数
                                                  groupSize,             // 高度
                                                  mChannelVolume         // 宽度
                                                  ));

    // 设置 cudnn batch normalization 描述符
    PLUGIN_CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(bnDesc, desc, CUDNN_BATCHNORM_SPATIAL));
    PLUGIN_CHECK_CUDNN(cudnnSetStream(_cudnn_handle, stream));

    PLUGIN_ASSERT(mBnScales && mBnScales->mPtr);
    PLUGIN_ASSERT(mBnBias && mBnBias->mPtr);

    float a = 1.F;
    float b = 0.F;
    PLUGIN_CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
        _cudnn_handle,           // cudnn 句柄
        CUDNN_BATCHNORM_SPATIAL, // 批量归一化模式，可以是 BATCHNORM_MODE_SPATIAL 或 BATCHNORM_MODE_PER_ACTIVATION
        &a,                      // 指向缩放因子的指针
        &b,                      // 指向偏移因子的指针
        desc,                    // 输入张量的描述符
        inputs[0],               // 输入张量的数据指针
        desc,                    // 输出张量的描述符
        outputs[0],              // 输出张量的数据指针
        bnDesc,                  // 批量归一化缩放、偏移、均值和方差的描述符
        mBnScales->mPtr,         // 批量归一化缩放因子的数据指针
        mBnBias->mPtr,           // 批量归一化偏移因子的数据指针
        0.0,                     // 用于计算滑动平均值和方差的指数平均值因子
        nullptr,                 // resultRunningMean 输出滑动平均值的数据指针
        nullptr,                 // resultRunningVar 输出滑动方差的数据指针
        mEpsilon,                // 用于计算方差和标准差的小常数
        nullptr, // resultSaveMean 用于缓存中间结果的缓冲区，可以在反向传播时加速。
        nullptr  // resultSaveInvVar 用于缓存中间结果的缓冲区，可以在反向传播时加速。
        ));

    float *output = static_cast<float *>(outputs[0]);
    PLUGIN_ASSERT(mGnBetas && mGnBetas->mPtr);
    PLUGIN_ASSERT(mGnGammas && mGnGammas->mPtr);
    return scaleShiftChannelsInplace(output, batchSize, mNbChannels, mChannelVolume, (float *)mGnBetas->mPtr,
                                     (float *)mGnGammas->mPtr, stream);
}

size_t GroupNormalizationPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNbGroups) + sizeof(mEpsilon) + sizeof(mNbScaleBias);
}

void GroupNormalizationPlugin::serialize(void *buffer) const noexcept
{
    serialize_value(&buffer, mEpsilon);
    serialize_value(&buffer, mNbGroups);
    serialize_value(&buffer, mNbChannels);
    serialize_value(&buffer, mNbScaleBias);
    // TODO: 序列化 gamma 和 beta 数据
}

bool GroupNormalizationPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
                                                         int nbOutputs) noexcept
{
    PLUGIN_ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
            && inOut[pos].type == inOut[0].type);
}

void GroupNormalizationPlugin::terminate() noexcept {}

void GroupNormalizationPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt *GroupNormalizationPlugin::clone() const noexcept
{
    try
    {
        auto *plugin = new GroupNormalizationPlugin(mEpsilon, mNbGroups, mNbChannels);
        plugin->setPluginNamespace(mPluginNamespace);
        plugin->mNbScaleBias = mNbScaleBias;
        plugin->mBnScales    = mBnScales;
        plugin->mBnBias      = mBnBias;
        plugin->mNbChannels  = mNbChannels;
        plugin->mGnBetas     = mGnBetas;
        plugin->mGnGammas    = mGnGammas;
        return plugin;
    }
    catch (const std::exception &e)
    {
        caughtError(e);
    }
    return nullptr;
}

void GroupNormalizationPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                               const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept
{
    const int32_t batchSize = in[0].desc.dims.d[0] <= 0 ? in[0].max.d[0] : in[0].desc.dims.d[0];
    mNbScaleBias            = batchSize * mNbGroups;
}

nvinfer1::DataType GroupNormalizationPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                               int nbInputs) const noexcept
{
    PLUGIN_ASSERT(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

size_t GroupNormalizationPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                                  const nvinfer1::PluginTensorDesc *outputs,
                                                  int                               nbOutputs) const noexcept
{
    return 0;
}

void GroupNormalizationPlugin::setPluginNamespace(const char *libNamespace) noexcept
{
    mPluginNamespace = libNamespace;
}

const char *GroupNormalizationPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

GroupNormalizationPluginCreator::GroupNormalizationPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_groups", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_channels", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

const char *GroupNormalizationPluginCreator::getPluginName() const noexcept
{
    return kGROUP_NORM_NAME;
}

const char *GroupNormalizationPluginCreator::getPluginVersion() const noexcept
{
    return kGROUP_NORM_VERSION;
}

const PluginFieldCollection *GroupNormalizationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

const char *GroupNormalizationPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GroupNormalizationPluginCreator::setPluginNamespace(const char *libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt *GroupNormalizationPluginCreator::createPlugin(const char                  *name,
                                                                   const PluginFieldCollection *fc) noexcept
{
    try
    {
        // Set default values
        int   nbGroups{1};
        float epsilon{0.00001F};
        int   nbChannels{1};
        for (int i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare("eps") == 0)
            {
                epsilon = *static_cast<const float *>(fc->fields[i].data);
            }
            if (field_name.compare("num_groups") == 0)
            {
                nbGroups = *static_cast<const int *>(fc->fields[i].data);
            }
            if (field_name.compare("num_channels") == 0)
            {
                nbChannels = *static_cast<const int *>(fc->fields[i].data);
            }
        }

        GroupNormalizationPlugin *plugin = new GroupNormalizationPlugin(epsilon, nbGroups, nbChannels);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    }
    catch (const std::exception &e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt *GroupNormalizationPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                                                        size_t serialLength) noexcept
{
    try
    {
        GroupNormalizationPlugin *plugin = new GroupNormalizationPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    }
    catch (const std::exception &e)
    {
        caughtError(e);
    }
    return nullptr;
}

// 调用GroupNormalizationPluginCreator返回一个IPluginV2Layer*
IPluginV2Layer *nvinfer1::plugin::addGroupNormLayer(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                                                    int num_groups, int num_channels, float epsilon)
{
    // 创建GroupNormalizationPlugin
    auto plugin_creator = getPluginRegistry()->getPluginCreator(kGROUP_NORM_NAME, kGROUP_NORM_VERSION);
    PLUGIN_ASSERT(plugin_creator);

    PluginField plugin_fields[3];
    plugin_fields[0].name   = "eps";
    plugin_fields[0].data   = &epsilon;
    plugin_fields[0].type   = PluginFieldType::kFLOAT32;
    plugin_fields[0].length = 1;

    plugin_fields[1].name   = "num_groups";
    plugin_fields[1].data   = &num_groups;
    plugin_fields[1].type   = PluginFieldType::kINT32;
    plugin_fields[1].length = 1;

    plugin_fields[2].name   = "num_channels";
    plugin_fields[2].data   = &num_channels;
    plugin_fields[2].type   = PluginFieldType::kINT32;
    plugin_fields[2].length = 1;

    // TODO: 从 weight 读取 gamma 和 beta 并传给 plugin

    PluginFieldCollection plugin_data;
    plugin_data.nbFields  = 3;
    plugin_data.fields    = plugin_fields;
    IPluginV2 *plugin_obj = plugin_creator->createPlugin(kGROUP_NORM_NAME, &plugin_data);

    PLUGIN_ASSERT(plugin_obj);
    nvinfer1::ITensor *input_tensors[] = {&input};
    // 这里要求三个输入 input、gamma、beta
    // gamma、beta 的大小等于 input 的通道数
    input.getDimensions();
    auto *plugin_layer = network->addPluginV2(input_tensors, 1, *plugin_obj);
    return plugin_layer;
}
