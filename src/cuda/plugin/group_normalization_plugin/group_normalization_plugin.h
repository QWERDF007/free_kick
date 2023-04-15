#pragma once
#include "common/plugin.h"

#include <cudnn.h>

#include <memory>
#include <string>
#include <vector>

namespace nvinfer1 { namespace plugin {

/**
 * @brief 对输入的张量进行归一化
 *
 * @tparam T
 * @param inOut 输入张量
 * @param B 批量大小
 * @param C 通道数
 * @param channelVolume 通道体积
 * @param beta beta参数
 * @param gamma gamma参数
 * @param stream cuda流
 * @return cudaError_t
 */
template<typename T>
cudaError_t scaleShiftChannelsInplace(T *inOut, const int B, const int C, const int channelVolume, float *beta,
                                      float *gamma, cudaStream_t stream);

class TRT_PLUGIN_API GroupNormalizationPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    /**
     * @brief 构造函数
     *
     * @param epsilon
     * @param nbGroups
     */
    GroupNormalizationPlugin(float epsilon, int nbGroups, int nbChannels);

    /**
     * @brief 从序列化数据构造插件
     *
     * @param data
     * @param length
     */
    GroupNormalizationPlugin(const void *data, size_t length);

    // It doesn't make sense to make GroupNormalizationPlugin without arguments,
    // so we delete default constructor.
    GroupNormalizationPlugin() = delete;

    /**
     * @brief 获取输出数量
     *
     * @return int
     */
    int getNbOutputs() const noexcept override;

    /**
     * @brief 获取输出维度
     *
     * @param index
     * @param inputs
     * @param nbInputDims
     * @param exprBuilder
     * @return DimsExprs
     */
    DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                  nvinfer1::IExprBuilder &exprBuilder) noexcept override;

    /**
     * @brief 初始化插件
     *
     * @return int
     */
    int initialize() noexcept override;

    /**
     * @brief 终止插件
     *
     */
    void terminate() noexcept override;

    /**
     * @brief 获取工作空间大小
     *
     * @param inputs
     * @param nbInputs
     * @param outputs
     * @param nbOutputs
     * @return size_t
     */
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                            const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept override;

    /**
     * @brief 执行插件
     *
     * @param inputDesc
     * @param outputDesc
     * @param inputs
     * @param outputs
     * @param workspace
     * @param stream
     * @return int
     */
    int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                const void *const *inputs, void *const *outputs, void *workspace,
                cudaStream_t stream) noexcept override;

    /**
     * @brief 获取序列化大小
     *
     * @return size_t
     */
    size_t getSerializationSize() const noexcept override;

    /**
     * @brief 序列化插件
     *
     * @param buffer
     */
    void serialize(void *buffer) const noexcept override;

    /**
     * @brief 是否支持输入输出格式组合
     *
     * @param pos
     * @param inOut
     * @param nbInputs
     * @param nbOutputs
     * @return bool
     */
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
                                   int nbOutputs) noexcept override;

    /**
     * @brief 获取插件类型
     *
     * @return char const*
     */
    const char *getPluginType() const noexcept override;

    /**
     * @brief 获取插件版本
     *
     * @return char const*
     */
    const char *getPluginVersion() const noexcept override;

    /**
     * @brief 克隆插件
     *
     * @return nvinfer1::IPluginV2DynamicExt*
     */
    nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;

    /**
     * @brief 销毁插件
     *
     */
    void destroy() noexcept override;

    /**
     * @brief 获取输出数据类型
     *
     * @param index
     * @param inputTypes
     * @param nbInputs
     * @return DataType
     */
    DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override;

    /**
     * @brief 绑定上下文
     *
     * @param cudnn
     * @param cublas
     * @param allocator
     */
    void attachToContext(cudnnContext *cudnn, cublasContext *cublas,
                         nvinfer1::IGpuAllocator *allocator) noexcept override;

    /**
     * @brief 解绑上下文
     *
     */
    void detachFromContext() noexcept override;

    /**
     * @brief 设置插件命名空间
     *
     * @param pluginNamespace
     */
    void setPluginNamespace(const char *pluginNamespace) noexcept override;

    /**
     * @brief 获取插件命名空间
     *
     * @return char const*
     */
    const char *getPluginNamespace() const noexcept override;

    /**
     * @brief 配置插件
     *
     * @param in
     * @param nbInputs
     * @param out
     * @param nbOutputs
     */
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept override;

private:
    const char *mPluginNamespace;
    std::string mNamespace;

    float mEpsilon;
    int   mNbGroups;
    int   mChannelVolume;

    cudnnHandle_t                    _cudnn_handle;
    // 描述输入和输出
    cudnnTensorDescriptor_t          desc;
    cudnnTensorDescriptor_t          bnDesc;
    // 这些缓冲区初始化为1和0
    std::shared_ptr<CudaBind<float>> mBnScales{};
    std::shared_ptr<CudaBind<float>> mBnBias{};
    std::shared_ptr<CudaBind<float>> mGnGammas{};
    std::shared_ptr<CudaBind<float>> mGnBetas{};
    size_t                           mNbScaleBias{};
    size_t                           mNbChannels{};

    using IPluginV2::enqueue;
    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2Ext::configurePlugin;
};

class GroupNormalizationPluginCreator : public IPluginCreator
{
public:
    /**
     * @brief 构造函数
     *
     */
    GroupNormalizationPluginCreator();

    /**
     * @brief 析构函数
     *
     */
    ~GroupNormalizationPluginCreator() override = default;

    /**
     * @brief 获取插件名称
     *
     * @return char const*
     */
    const char *getPluginName() const noexcept override;

    /**
     * @brief 获取插件版本
     *
     * @return char const*
     */
    const char *getPluginVersion() const noexcept override;

    /**
     * @brief 获取插件属性
     *
     * @return PluginFieldCollection const*
     */
    const PluginFieldCollection *getFieldNames() noexcept override;

    /**
     * @brief 创建插件
     *
     * @param name
     * @param fc
     * @return IPluginV2DynamicExt*
     */
    IPluginV2DynamicExt *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;

    /**
     * @brief 反序列化插件
     *
     * @param name
     * @param serialData
     * @param serialLength
     * @return IPluginV2DynamicExt*
     */
    IPluginV2DynamicExt *deserializePlugin(const char *name, const void *serialData,
                                           size_t serialLength) noexcept override;

    /**
     * @brief 设置插件命名空间
     *
     * @param pluginNamespace
     */
    void setPluginNamespace(const char *pluginNamespace) noexcept override;

    /**
     * @brief 获取插件命名空间
     *
     * @return char const*
     */
    const char *getPluginNamespace() const noexcept override;

private:
    // PluginFieldCollection是一个类，它是用于存储插件字段的集合。它是Sitecore的一个类，用于存储插件字段的集合。12
    static PluginFieldCollection    mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string                     mNamespace;
};

// 调用GroupNormalizationPluginCreator返回一个IPluginV2Layer*
TRT_PLUGIN_API IPluginV2Layer *addGroupNormLayer(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                                                 int num_groups, int num_channels, float epsilon = 1e-5);

}} // namespace nvinfer1::plugin
