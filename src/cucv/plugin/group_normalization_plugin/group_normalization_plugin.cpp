#include "group_normalization_plugin.h"
#include "common/dimsHelpers.h"
#include "common/serialize.hpp"

using namespace nvinfer1;
using nvinfer1::plugin::GroupNormalizationPlugin;
using nvinfer1::plugin::GroupNormalizationPluginCreator;

namespace {
constexpr char const *kGROUP_NORM_VERSION{"1"};                     // GroupNormalizationPlugin版本号
constexpr char const *kGROUP_NORM_NAME{"GroupNormalizationPlugin"}; // GroupNormalizationPlugin名称
} // namespace

// // Static class fields initialization
PluginFieldCollection GroupNormalizationPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GroupNormalizationPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GroupNormalizationPluginCreator);

GroupNormalizationPlugin::GroupNormalizationPlugin(float epsilon, int nbGroups)
    : mEpsilon(epsilon) // 初始化mEpsilon为epsilon
      ,
      mNbGroups(nbGroups) // 初始化mNbGroups为nbGroups
{
    PLUGIN_VALIDATE(mEpsilon > 0.0F); // 确保mEpsilon大于0
    // Number of groups should be positive
    PLUGIN_VALIDATE(mNbGroups > 0); // 确保mNbGroups大于0
}

int GroupNormalizationPlugin::initialize() noexcept {
    auto allocScaleBias = [this](std::shared_ptr<CudaBind<float>> &buf, float value) {
        PLUGIN_VALIDATE(mNbScaleBias > 0); // 确保mNbScaleBias大于0
        if (!buf || !buf->mPtr || buf->mSize != mNbScaleBias) {
            // Allocate device memory.
            buf = std::make_shared<CudaBind<float>>(mNbScaleBias); // 分配设备内存

            // Initialize values.
            std::vector<float> const values(mNbScaleBias, value); // 初始化values为长度为mNbScaleBias，值为value的向量
            PLUGIN_CUASSERT(cudaMemcpy(buf->mPtr, values.data(), sizeof(float) * mNbScaleBias,
                                       cudaMemcpyHostToDevice)); // 将values拷贝到buf->mPtr中
        }
    };

    allocScaleBias(mBnScales, 1.F); // 初始化mBnScales
    allocScaleBias(mBnBias, 0.F);   // 初始化mBnBias
    return 0;
}

GroupNormalizationPlugin::GroupNormalizationPlugin(void const *data, size_t length) {
    // 反序列化，按照序列化的顺序
    deserialize_value(&data, &length, &mEpsilon);     // 反序列化mEpsilon
    deserialize_value(&data, &length, &mNbGroups);    // 反序列化mNbGroups
    deserialize_value(&data, &length, &mNbScaleBias); // 反序列化mNbScaleBias
}

char const *GroupNormalizationPlugin::getPluginType() const noexcept {
    return kGROUP_NORM_NAME; // 返回GroupNormalizationPlugin名称
}

char const *GroupNormalizationPlugin::getPluginVersion() const noexcept {
    return kGROUP_NORM_VERSION; // 返回GroupNormalizationPlugin版本号
}

int GroupNormalizationPlugin::getNbOutputs() const noexcept {
    return 1; // 返回输出数量为1
}

nvinfer1::DimsExprs GroupNormalizationPlugin::getOutputDimensions(int index, nvinfer1::DimsExprs const *inputs,
                                                                  int nbInputs,
                                                                  nvinfer1::IExprBuilder &exprBuilder) noexcept {
    // 插件的三个输入分别为上一层的输入、gamma 和 beta
    PLUGIN_ASSERT(nbInputs == 3);          // 确保输入数量为3
    PLUGIN_ASSERT(index == 0);             // 确保输出索引为0
    nvinfer1::DimsExprs output(inputs[0]); // 输出的维度与上一层的输入相同
    return output;                         // 返回输出的维度
}

void GroupNormalizationPlugin::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                                               IGpuAllocator *gpuAllocator) noexcept {
    PLUGIN_ASSERT(cudnnContext);                              // 确保cudnnContext不为空
    _cudnn_handle = cudnnContext;                             // 将cudnnContext赋值给_cudnn_handle
    PLUGIN_CUDNNASSERT(cudnnCreateTensorDescriptor(&desc));   // 创建描述符desc
    PLUGIN_CUDNNASSERT(cudnnCreateTensorDescriptor(&bnDesc)); // 创建描述符bnDesc
}

// 将插件对象从其执行上下文中分离
void GroupNormalizationPlugin::detachFromContext() noexcept {
    PLUGIN_CUDNNASSERT(cudnnDestroyTensorDescriptor(desc));   // 销毁描述符desc
    PLUGIN_CUDNNASSERT(cudnnDestroyTensorDescriptor(bnDesc)); // 销毁描述符bnDesc
}

int GroupNormalizationPlugin::enqueue(nvinfer1::PluginTensorDesc const *inputDesc,
                                      nvinfer1::PluginTensorDesc const *outputDesc, void const *const *inputs,
                                      void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
    // 获取输入维度
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    int batchSize = input_dims.d[0];
    int nbChannels = input_dims.d[1];

    // 计算每个组的大小
    int groupSize = nbChannels / mNbGroups;

    // 计算每个通道的体积
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

    // 根据 cudnnSetTensor4dDescriptor 重塑数据
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
    return scaleShiftChannelsInplace(output, batchSize, nbChannels, mChannelVolume,
                                     static_cast<float const *>(inputs[2]), static_cast<float const *>(inputs[1]),
                                     stream); // mBetaDev, mGammaDev,
}

size_t GroupNormalizationPlugin::getSerializationSize() const noexcept {
    return sizeof(mNbGroups) + sizeof(mEpsilon) + sizeof(mNbScaleBias);
}

void GroupNormalizationPlugin::serialize(void *buffer) const noexcept {
    serialize_value(&buffer, mEpsilon);
    serialize_value(&buffer, mNbGroups);
    serialize_value(&buffer, mNbScaleBias);
}

bool GroupNormalizationPlugin::supportsFormatCombination(int pos, nvinfer1::PluginTensorDesc const *inOut, int nbInputs,
                                                         int nbOutputs) noexcept {
    PLUGIN_ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR &&
            inOut[pos].type == inOut[0].type);
}

void GroupNormalizationPlugin::terminate() noexcept {}

void GroupNormalizationPlugin::destroy() noexcept {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt *GroupNormalizationPlugin::clone() const noexcept {
    try {
        auto *plugin = new GroupNormalizationPlugin(mEpsilon, mNbGroups);
        plugin->setPluginNamespace(mPluginNamespace);
        plugin->mNbScaleBias = mNbScaleBias;
        plugin->mBnScales = mBnScales;
        plugin->mBnBias = mBnBias;
        return plugin;
    } catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

void GroupNormalizationPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const *in, int nbInputs,
                                               nvinfer1::DynamicPluginTensorDesc const *out, int nbOutputs) noexcept {
    int32_t const batchSize = in[0].desc.dims.d[0] <= 0 ? in[0].max.d[0] : in[0].desc.dims.d[0];
    mNbScaleBias = batchSize * mNbGroups;
}

nvinfer1::DataType GroupNormalizationPlugin::getOutputDataType(int index, nvinfer1::DataType const *inputTypes,
                                                               int nbInputs) const noexcept {
    PLUGIN_ASSERT(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

size_t GroupNormalizationPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const *inputs, int nbInputs,
                                                  nvinfer1::PluginTensorDesc const *outputs,
                                                  int nbOutputs) const noexcept {
    return 0;
}

void GroupNormalizationPlugin::setPluginNamespace(char const *libNamespace) noexcept {
    mPluginNamespace = libNamespace;
}

char const *GroupNormalizationPlugin::getPluginNamespace() const noexcept { return mPluginNamespace; }

GroupNormalizationPluginCreator::GroupNormalizationPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_groups", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const *GroupNormalizationPluginCreator::getPluginName() const noexcept { return kGROUP_NORM_NAME; }

char const *GroupNormalizationPluginCreator::getPluginVersion() const noexcept { return kGROUP_NORM_VERSION; }

PluginFieldCollection const *GroupNormalizationPluginCreator::getFieldNames() noexcept { return &mFC; }

char const *GroupNormalizationPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

void GroupNormalizationPluginCreator::setPluginNamespace(char const *libNamespace) noexcept {
    mNamespace = libNamespace;
}

IPluginV2DynamicExt *GroupNormalizationPluginCreator::createPlugin(char const *name,
                                                                   PluginFieldCollection const *fc) noexcept {
    try {
        // Set default values
        int nbGroups{1};
        float epsilon{0.00001F};
        for (int i = 0; i < fc->nbFields; i++) {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare("eps") == 0) {
                epsilon = *static_cast<float const *>(fc->fields[i].data);
            }
            if (field_name.compare("num_groups") == 0) {
                nbGroups = *static_cast<int const *>(fc->fields[i].data);
            }
        }

        GroupNormalizationPlugin *plugin = new GroupNormalizationPlugin(epsilon, nbGroups);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    } catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt *GroupNormalizationPluginCreator::deserializePlugin(char const *name, void const *serialData,
                                                                        size_t serialLength) noexcept {
    try {
        GroupNormalizationPlugin *plugin = new GroupNormalizationPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    } catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

// 调用GroupNormalizationPluginCreator返回一个IPluginV2Layer*
IPluginV2Layer *nvinfer1::plugin::addGroupNormLayer(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                                                    int num_groups, float epsilon) {

    // 创建GroupNormalizationPlugin
    std::cout << "Create plugin creator" << std::endl;
    auto plugin_creator = getPluginRegistry()->getPluginCreator(kGROUP_NORM_NAME, kGROUP_NORM_VERSION);
    std::cout << "plugin_creator ptr: " << (void *)plugin_creator << std::endl;
    std::cout << "Create PluginField" << std::endl;
    PluginField plugin_fields[2];
    plugin_fields[0].name = "eps";
    plugin_fields[0].data = &epsilon;
    plugin_fields[0].type = PluginFieldType::kFLOAT32;
    plugin_fields[0].length = 1;

    plugin_fields[1].name = "num_groups";
    plugin_fields[1].data = &num_groups;
    plugin_fields[1].type = PluginFieldType::kINT32;
    plugin_fields[1].length = 1;

    std::cout << "Create PluginFieldCollection" << std::endl;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    std::cout << "Create plugin" << std::endl;
    IPluginV2 *plugin_obj = plugin_creator->createPlugin(kGROUP_NORM_NAME, &plugin_data);
    std::cout << "plugin_obj ptr: " << (void *)plugin_obj << std::endl;
    if (plugin_creator == nullptr) {
        return nullptr;
    }
    std::cout << "addPluginV2" << std::endl;
    // plugin_creator
    nvinfer1::ITensor *input_tensors[] = {&input};
    auto *plugin_layer = network->addPluginV2(input_tensors, 1, *plugin_obj);
    std::cout << "addPluginV2 done" << std::endl;
    std::cout << "plugin_layer ptr: " << (void *)plugin_layer << std::endl;
    return plugin_layer;
}
