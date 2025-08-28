//
//  llm.cpp
//  大语言模型(LLM)核心实现文件
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//
// #define MNN_OPEN_TIME_TRACE 1

#include <fstream>       // 文件流操作
#include <iostream>      // 标准输入输出  
#include <regex>         // 正则表达式处理
#include <sstream>       // 字符串流操作
#include <unordered_set> // 无序集合容器

#include <MNN/AutoTime.hpp>               // MNN自动计时工具
#include <MNN/expr/ExecutorScope.hpp>     // MNN执行器作用域管理
#include "cpp/ExprDebug.hpp"              // MNN表达式调试工具
#include "llm.hpp"                        // LLM主类定义
#include "llmconfig.hpp"                  // LLM配置管理类
#include "tokenizer.hpp"                  // 分词器类定义
#include "logger.hpp"                     // 文件日志系统
// 调试模式设置: 0: 无调试, 1: 测试操作时间, 2: 打印张量信息, 3: 打印输出张量
#define DEBUG_MODE 0

#include "httplib.h"                      // HTTP客户端库，用于网络请求
#ifdef LLM_SUPPORT_VISION
#include <cv/cv.hpp>  // 视觉支持相关头文件
#endif
#ifdef LLM_SUPPORT_AUDIO
#include <audio/audio.hpp>  // 音频支持相关头文件
#endif

using namespace MNN::Express;  // 使用MNN表达式命名空间
namespace MNN {
namespace Transformer {
/**
 * @brief KV缓存元数据结构，用于管理键值对缓存的状态和内存分配
 * 
 * 该结构体负责跟踪和管理Transformer模型中注意力机制的KV缓存状态，
 * 包括缓存的增删改操作以及内存块的管理。
 */
struct KVMeta {
    size_t block = 4096;      ///< 内存块大小，默认4096字节，用于内存块管理
    size_t previous = 0;      ///< 之前的序列长度，记录历史token数量
    size_t remove = 0;        ///< 需要移除的token数量，用于缓存清理
    int* reserve = nullptr;   ///< 保留区域指针，指向需要保留的内存区域
    int n_reserve = 0;        ///< 保留区域数量
    size_t add = 0;           ///< 新增的token数量
    std::vector<int> reserveHost;  ///< 主机端保留数据的缓存
    
    /**
     * @brief 同步缓存状态，更新序列长度并重置临时变量
     * 
     * 该方法计算并更新当前KV缓存的总序列长度，考虑了移除、新增和恢复的token数量，
     * 然后重置所有临时状态变量，为下一次操作做准备。
     */
    void sync() {
        int revertNumber = 0;
        // 遍历保留区域，累计需要恢复的token数量
        for (int i=0; i<n_reserve; ++i) {
            revertNumber += reserve[2*i+1];  // 每个保留区域的第二个值表示恢复数量
        }
        // 更新总的序列长度：之前的长度 - 移除数量 + 新增数量 + 恢复数量
        previous = previous - remove + add + revertNumber;
        // 重置临时状态变量
        n_reserve = 0;
        reserve = nullptr;
        remove = 0;
        add = 0;
    }
};
// 反量化函数类型定义：将量化的uint8数据转换为float数据
typedef void (*DequantFunction)(const uint8_t*, float*, float, float, int);

// 4位量化反量化函数（参考实现）
// 将4位量化的数据转换回float类型
static void q41_dequant_ref(const uint8_t* src, float* dst, float scale, float zero, int size) {
    for (int i = 0; i < size / 2; i++) {
        int x          = src[i];         // 读取一个字节，包含两个4位数值
        int x1         = x / 16 - 8;     // 提取高4位并减去偏移量8
        int x2         = x % 16 - 8;     // 提取低4位并减去偏移量8
        float w1       = x1 * scale + zero;  // 反量化第一个值
        float w2       = x2 * scale + zero;  // 反量化第二个值
        dst[2 * i]     = w1;             // 存储第一个反量化值
        dst[2 * i + 1] = w2;             // 存储第二个反量化值
    }
}

// 8位量化反量化函数（参考实现）
// 将8位量化的数据转换回float类型
static void q81_dequant_ref(const uint8_t* src, float* dst, float scale, float zero, int size) {
    for (int i = 0; i < size; i++) {
        // 8位量化: 减去128（无符号转有符号），然后应用缩放和零点
        dst[i] = (src[i] - 128) * scale + zero;
    }
}

/**
 * @brief 磁盘嵌入类：用于从磁盘文件中读取词向量嵌入，以节省内存使用
 * 
 * 该类支持量化和非量化两种存储格式，能够动态加载需要的嵌入向量，
 * 从而有效减少内存占用。对于大型模型，将词嵌入存储在磁盘上而不是内存中
 * 可以显著降低内存使用量。
 */
class DiskEmbedding {
public:
    /**
     * @brief 构造函数：根据配置文件初始化磁盘嵌入
     * 
     * 根据配置信息判断是否使用量化存储格式，并相应地初始化文件指针、
     * 缓冲区和反量化函数。
     * 
     * @param config 模型配置指针，包含嵌入层相关配置信息
     */
    explicit DiskEmbedding(const std::shared_ptr<LlmConfig>& config);
    
    /**
     * @brief 析构函数：清理资源
     * 
     * 使用默认析构函数，智能指针会自动释放相关资源。
     */
    ~DiskEmbedding() = default;
    
    /**
     * @brief 根据输入的token ID序列生成对应的嵌入向量
     * 
     * 该方法根据token ID从磁盘文件中读取对应的嵌入向量数据，
     * 如果是量化格式则进行反量化处理，最终将结果存储到指定的内存位置。
     * 
     * @param input_ids token ID序列
     * @param ptr 输出嵌入向量的指针，用于存储生成的嵌入向量
     */
    void embedding(const std::vector<int>& input_ids, float* ptr);

private:
    /**
     * @brief 从文件指定偏移位置读取数据到目标缓冲区
     * 
     * 该方法将文件指针移动到指定偏移位置，并读取指定大小的数据到目标缓冲区。
     * 
     * @param dst 目标缓冲区指针
     * @param size 要读取的数据大小（字节数）
     * @param offset 文件中的偏移位置
     */
    void seek_read(uint8_t* dst, size_t size, size_t offset);
    
    std::unique_ptr<uint8_t[]> alpha_;   ///< 量化参数缓冲区（缩放因子和零点）
    std::unique_ptr<uint8_t[]> weight_;  ///< 权重数据缓冲区
    std::unique_ptr<FILE, decltype(&fclose)> fp_;  ///< 文件指针，用RAII管理
    DequantFunction dequant_;            ///< 反量化函数指针
    int hidden_size_;                    ///< 隐藏层维度（嵌入向量维度）
    int weight_token_size_;              ///< 每个token权重大小（字节数）
    int64_t w_offset_;                   ///< 权重在文件中的偏移位置
    int64_t block_num_;                  ///< 块数量（用于量化）
    int64_t quant_block_;                ///< 量化块大小
    int64_t quant_bit_;                  ///< 量化位数（4位或8位）
};

// 从文件指定偏移位置读取数据到目标缓冲区
void DiskEmbedding::seek_read(uint8_t* dst, size_t size, size_t offset) {
    fseek(fp_.get(), offset, SEEK_SET);  // 移动文件指针到指定偏移位置
    size_t bytes_read = fread(dst, 1, size, fp_.get());  // 读取指定大小的数据
    (void)bytes_read;  // 避免未使用变量警告
}

// DiskEmbedding构造函数：初始化磁盘嵌入系统
DiskEmbedding::DiskEmbedding(const std::shared_ptr<LlmConfig>& config) : fp_(nullptr, &fclose) {
    auto tie_embeddings = config->tie_embeddings();  // 获取绑定嵌入配置
    hidden_size_        = config->hidden_size();     // 获取隐藏层维度
    
    if (tie_embeddings.size() == 5) {
        // 量化嵌入模式：从主权重文件中读取量化的嵌入数据
        w_offset_          = tie_embeddings[0];  // 权重在文件中的偏移
        quant_bit_         = tie_embeddings[3];  // 量化位数（4位或8位）
        quant_block_       = tie_embeddings[4];  // 量化块大小
        block_num_         = hidden_size_ / quant_block_;  // 计算块数量
        weight_token_size_ = hidden_size_ * quant_bit_ / 8;  // 每个token的权重字节数
        fp_.reset(fopen(config->llm_weight().c_str(), "rb"));  // 打开权重文件
        
        // TODO: 优化反量化函数选择
        dequant_        = quant_bit_ == 8 ? q81_dequant_ref : q41_dequant_ref;  // 选择对应的反量化函数
        auto a_offset   = tie_embeddings[1];  // alpha参数偏移
        auto alpha_size = tie_embeddings[2];  // alpha参数大小
        alpha_.reset(new uint8_t[alpha_size]);  // 分配alpha参数缓冲区
        seek_read(alpha_.get(), alpha_size, a_offset);  // 读取alpha参数
    } else {
        // 非量化嵌入模式：使用专门的嵌入文件（bf16格式）
        weight_token_size_ = hidden_size_ * sizeof(int16_t);  // bf16格式，每个元素2字节
        fp_.reset(fopen(config->embedding_file().c_str(), "rb"));  // 打开嵌入文件
    }
    weight_.reset(new uint8_t[weight_token_size_]);  // 分配权重缓冲区
}

// 根据token ID序列生成对应的嵌入向量
void DiskEmbedding::embedding(const std::vector<int>& input_ids, float* dst) {
    if (alpha_.get()) {
        // 量化模式：需要读取量化数据并反量化
        for (size_t i = 0; i < input_ids.size(); i++) {
            int token = input_ids[i];  // 当前token ID
            // 读取该token对应的量化权重数据
            seek_read(weight_.get(), weight_token_size_, w_offset_ + token * weight_token_size_);
            auto dptr      = dst + i * hidden_size_;  // 目标输出位置
            auto alpha_ptr = reinterpret_cast<float*>(alpha_.get()) + token * block_num_ * 2;  // alpha参数位置
            
            // 按块进行反量化
            for (int n = 0; n < block_num_; n++) {
                auto dst_ptr     = dptr + n * quant_block_;  // 当前块的输出位置
                uint8_t* src_ptr = weight_.get() + n * (quant_block_ * quant_bit_ / 8);  // 当前块的输入位置
                float zero       = (alpha_ptr + n * 2)[0];    // 零点参数
                float scale      = (alpha_ptr + n * 2)[1];    // 缩放因子参数
                dequant_(src_ptr, dst_ptr, scale, zero, quant_block_);  // 执行反量化
            }
        }
    } else {
        // bf16模式：直接读取bf16数据并转换为float
        for (size_t i = 0; i < input_ids.size(); i++) {
            // 读取该token对应的bf16权重数据
            seek_read(weight_.get(), weight_token_size_, input_ids[i] * weight_token_size_);
            int16_t* dst_ptr = reinterpret_cast<int16_t*>(dst + i * hidden_size_);
            
            // 将bf16格式转换为float：bf16在内存中是高16位，低16位为0
            for (int j = 0; j < hidden_size_; j++) {
                dst_ptr[j * 2]     = 0;  // 低16位设为0
                dst_ptr[j * 2 + 1] = reinterpret_cast<int16_t*>(weight_.get())[j];  // 高16位来自文件
            }
        }
    }
}

/**
 * @brief 多模态大语言模型类：继承自Llm，支持视觉和音频输入处理
 * 
 * 该类扩展了基础的Llm类，增加了对图像和音频数据的处理能力。
 * 能够将图像和音频数据转换为token序列，并与文本一起输入模型进行多模态推理。
 */
class Mllm : public Llm {
public:
    /**
     * @brief 构造函数：初始化多模态LLM，配置视觉和音频参数
     * 
     * 根据配置文件初始化多模态LLM的相关参数，包括视觉和音频处理所需的配置。
     * 如果配置启用了视觉或音频功能，则相应地设置处理参数。
     * 
     * @param config 模型配置指针，包含多模态相关的配置信息
     */
    Mllm(std::shared_ptr<LlmConfig> config) : Llm(config) {
        if (config->is_visual()) {
            // 配置视觉相关参数
            image_height_  = config->llm_config_.value("image_size", image_height_);   // 图像高度
            image_width_   = image_height_;                                           // 图像宽度（通常为正方形）
            img_pad_       = config->llm_config_.value("image_pad", img_pad_);        // 图像填充token ID
            vision_start_  = config->llm_config_.value("vision_start", vision_start_); // 视觉序列开始token
            vision_end_    = config->llm_config_.value("vision_end", vision_end_);     // 视觉序列结束token
            image_mean_    = config->llm_config_.value("image_mean", image_mean_);     // 图像均值（用于归一化）
            image_norm_    = config->llm_config_.value("image_norm", image_norm_);     // 图像标准差（用于归一化）
        }
        if (config->is_audio()) {
            // 音频配置预留（待实现）
        }
    }
    
    /**
     * @brief 析构函数：清理多模态模块资源
     * 
     * 释放多模态处理模块的资源，确保正确清理内存。
     */
    ~Mllm() {
        mul_module_.reset();  // 释放多模态模块
    }
    
    /**
     * @brief 重写基类方法：加载多模态模型
     * 
     * 该方法扩展了基类的加载功能，除了加载基础的LLM模型外，
     * 还会加载多模态处理模块（如视觉编码器或音频编码器）。
     */
    virtual void load() override;
    
    /**
     * @brief 重写基类方法：支持多模态内容的token编码
     * 
     * 该方法扩展了基类的token编码功能，能够处理包含图像或音频标记的文本，
     * 将这些标记转换为对应的token序列。
     * 
     * @param query 输入的查询字符串，可能包含多模态标记
     * @param use_template 是否使用提示词模板，默认为true
     * @return 编码后的token ID序列
     */
    virtual std::vector<int> tokenizer_encode(const std::string& query, bool use_template = true) override;
    
    /**
     * @brief 重写基类方法：生成包含多模态内容的嵌入向量
     * 
     * 该方法扩展了基类的嵌入生成功能，能够处理包含多模态内容的token序列，
     * 将文本token和多模态token（图像或音频）一起转换为嵌入向量。
     * 
     * @param input_ids 输入的token ID序列，可能包含多模态token
     * @return 生成的嵌入向量
     */
    virtual MNN::Express::VARP embedding(const std::vector<int>& input_ids) override;

private:
    // 视觉配置参数
    int image_height_ = 448;                                       ///< 输入图像高度
    int image_width_ = 448;                                        ///< 输入图像宽度
    int vision_start_ = 151857;                                    ///< 视觉序列开始token ID
    int vision_end_ = 151858;                                      ///< 视觉序列结束token ID
    int img_pad_ = 151859;                                         ///< 图像填充token ID
    std::vector<float> image_mean_{122.7709383, 116.7460125, 104.09373615}; ///< RGB图像均值，用于归一化
    std::vector<float> image_norm_{0.01459843, 0.01500777, 0.01422007};     ///< RGB图像标准差，用于归一化
    
    // 音频配置参数
    int audio_pad_ = 151646;  ///< 音频填充token ID
    
    // 私有方法声明
    /**
     * @brief 多模态数据处理
     * 
     * 根据指定的模式（图像或音频）和文件信息，处理多模态数据并生成对应的token序列。
     * 
     * @param mode 处理模式，"img"表示图像，"audio"表示音频
     * @param info 文件信息，可能包含文件路径或URL
     * @return 处理后的token ID序列
     */
    std::vector<int> multimode_process(const std::string& mode, std::string info);
    
    /**
     * @brief 视觉数据处理
     * 
     * 处理图像文件，将其转换为token序列。支持多种视觉模型架构，
     * 包括标准模型和Qwen2-VL等。
     * 
     * @param file 图像文件路径
     * @return 图像对应的token ID序列
     */
    std::vector<int> vision_process(const std::string& file);
    
    /**
     * @brief 音频数据处理
     * 
     * 处理音频文件，将其转换为token序列。使用Whisper的filter bank特征提取和音频编码器。
     * 
     * @param file 音频文件路径
     * @return 音频对应的token ID序列
     */
    std::vector<int> audio_process(const std::string& file);
    
    // 多模态模块和嵌入缓存
    std::shared_ptr<Module> mul_module_;   ///< 多模态处理模块（视觉或音频编码器）
    std::vector<VARP> mul_embeddings_;     ///< 多模态嵌入向量缓存
};

// === Llm 类实现开始 ===

/**
 * @brief 静态工厂方法：根据配置文件路径创建LLM实例
 * 
 * 该方法是一个静态工厂方法，用于根据配置文件路径创建相应的LLM实例。
 * 自动判断是否需要多模态支持，如果配置启用了视觉或音频功能，
 * 则返回Mllm实例，否则返回标准的Llm实例。
 * 
 * @param config_path 配置文件路径
 * @return 创建的LLM实例指针
 */
Llm* Llm::createLLM(const std::string& config_path) {
    LOG_INFO("开始创建LLM实例，配置文件路径: " + config_path);
    
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));  // 解析配置文件
    Llm* llm = nullptr;
    
    // 根据配置判断是否需要多模态支持
    if (config->is_visual() || config->is_audio()) {
        LOG_INFO("检测到多模态配置，创建多模态LLM实例(Mllm)");
        llm = new Mllm(config);  // 创建多模态LLM实例
    } else {
        LOG_INFO("创建标准LLM实例");
        llm = new Llm(config);   // 创建标准LLM实例
    }
    
    LOG_INFO("LLM实例创建完成");
    return llm;
}

/**
 * @brief 后端类型转换函数：将字符串类型转换为MNN后端枚举
 * 
 * 该函数将表示后端类型的字符串转换为MNN的后端类型枚举值。
 * 支持多种后端类型，包括CPU、GPU和专用硬件加速器。
 * 
 * @param type_str 后端类型字符串
 * @return 对应的MNN后端类型枚举值
 */
static MNNForwardType backend_type_convert(const std::string& type_str) {
    if (type_str == "cpu")        return MNN_FORWARD_CPU;      // CPU后端
    if (type_str == "metal")      return MNN_FORWARD_METAL;    // Metal后端（iOS/macOS GPU）
    if (type_str == "cuda")       return MNN_FORWARD_CUDA;     // CUDA后端（NVIDIA GPU）
    if (type_str == "opencl")     return MNN_FORWARD_OPENCL;   // OpenCL后端（通用GPU）
    if (type_str == "opengl")     return MNN_FORWARD_OPENGL;   // OpenGL后端
    if (type_str == "vulkan")     return MNN_FORWARD_VULKAN;   // Vulkan后端
    if (type_str == "npu")        return MNN_FORWARD_NN;       // NPU后端（神经处理单元）
    return MNN_FORWARD_AUTO;      // 自动选择后端
}

/**
 * @brief 导出配置信息为JSON字符串
 * 
 * 该方法将当前模型的配置信息导出为JSON格式的字符串，
 * 便于查看和保存模型的配置状态。
 * 
 * @return JSON格式的配置信息字符串
 */
std::string Llm::dump_config() {
    return config_->config_.dump();
}

/**
 * @brief 设置配置信息：合并新的JSON配置
 * 
 * 该方法用于合并新的JSON配置信息到当前配置中，
 * 可以动态更新模型的配置参数。
 * 
 * @param content JSON格式的配置信息字符串
 * @return 设置成功返回true，否则返回false
 */
bool Llm::set_config(const std::string& content) {
    return config_->config_.merge(content.c_str());
}

/**
 * @brief 获取文件大小（以MB为单位）
 * 
 * 该函数用于获取指定文件的大小，并以MB为单位返回。
 * 主要用于计算模型文件的大小，为内存映射等操作提供参考。
 * 
 * @param filename 文件路径
 * @return 文件大小（MB），获取失败返回-1
 */
int file_size_m(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);  // 打开文件并移到末尾
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return -1;
    }
    long long fileSize = file.tellg();  // 获取文件大小（字节）
    file.close();
    return fileSize / (1024 * 1024);    // 转换为MB
}

/**
 * @brief 初始化运行时环境：配置MNN执行器和各种优化选项
 * 
 * 该方法用于初始化模型运行时环境，配置MNN执行器的各种参数和优化选项。
 * 包括后端类型、线程数、功耗模式、内存使用策略、精度模式等，
 * 以及各种运行时优化选项如量化、内存映射等。
 */
void Llm::init_runtime() {
    LOG_INFO("开始初始化运行时环境");
    
    // 配置调度和后端参数
    ScheduleConfig config;
    BackendConfig cpuBackendConfig;
    config.type      = backend_type_convert(config_->backend_type());  // 设置后端类型
    config.numThread = config_->thread_num();                          // 设置线程数
    
    LOG_INFO("配置后端类型: " + config_->backend_type() + 
             ", 线程数: " + std::to_string(config.numThread));
    
    // 设置全局执行器配置
    ExecutorScope::Current()->setGlobalExecutorConfig(config.type, cpuBackendConfig, config.numThread);
    
    // 配置功耗模式
    if (config_->power() == "high") {
        cpuBackendConfig.power = BackendConfig::Power_High;      // 高性能模式
        LOG_INFO("设置为高性能模式");
    } else if (config_->power() == "low") {
        cpuBackendConfig.power = BackendConfig::Power_Low;       // 低功耗模式
        LOG_INFO("设置为低功耗模式");
    }
    
    // 配置内存使用策略
    if (config_->memory() == "high") {
        cpuBackendConfig.memory = BackendConfig::Memory_High;    // 高内存使用（更多缓存）
        LOG_INFO("设置为高内存使用策略");
    } else if (config_->memory() == "low") {
        cpuBackendConfig.memory = BackendConfig::Memory_Low;     // 低内存使用（节省内存）
        LOG_INFO("设置为低内存使用策略");
    }
    
    // 配置精度模式
    if (config_->precision() == "high") {
        cpuBackendConfig.precision = BackendConfig::Precision_High;  // 高精度计算
        LOG_INFO("设置为高精度计算模式");
    } else if (config_->precision() == "low") {
        cpuBackendConfig.precision = BackendConfig::Precision_Low;   // 低精度计算（更快）
        LOG_INFO("设置为低精度计算模式");
    }
    
    config.backendConfig = &cpuBackendConfig;

    // 创建运行时管理器
    LOG_DEBUG("创建运行时管理器");
    runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
    
    // 设置各种运行时优化选项
    runtime_manager_->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);              // 内存分配器类型
    runtime_manager_->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, 1);           // 动态量化选项：1=按批次量化, 2=按张量量化
    runtime_manager_->setHint(MNN::Interpreter::QKV_QUANT_OPTIONS, config_->quant_qkv());     // QKV量化选项
    runtime_manager_->setHint(MNN::Interpreter::KVCACHE_SIZE_LIMIT, config_->kvcache_limit()); // KV缓存大小限制
    runtime_manager_->setHint(MNN::Interpreter::MMAP_FILE_SIZE, file_size_m(config_->llm_weight()) + 128); // 内存映射文件大小
    runtime_manager_->setHint(MNN::Interpreter::USE_CACHED_MMAP, 1);                 // 使用缓存的内存映射
    
    LOG_DEBUG("设置运行时优化选项: KV缓存限制=" + std::to_string(config_->kvcache_limit()) + 
              ", QKV量化=" + std::to_string(config_->quant_qkv()));
    
    std::string tmpPath = config_->tmp_path();
    
    // 配置KV缓存内存映射路径
    if (config_->kvcache_mmap()) {
        runtime_manager_->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
        LOG_INFO("启用KV缓存内存映射，路径: " + tmpPath);
    }
    
    // 配置权重内存映射路径
    if (config_->use_mmap()) {
        runtime_manager_->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
        LOG_INFO("启用权重内存映射，路径: " + tmpPath);
    }
    
    // 设置KV缓存元数据指针
    runtime_manager_->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());

    // 调试模式配置
#if DEBUG_MODE == 1
    runtime_manager_->setMode(MNN::Interpreter::Session_Debug);  // 启用调试模式
    _initTimeTrace();                                            // 初始化时间追踪
    LOG_DEBUG("启用调试模式: 时间追踪");
#endif
#if DEBUG_MODE == 2
    runtime_manager_->setMode(MNN::Interpreter::Session_Debug);  // 启用调试模式
    _initTensorStatic();                                         // 初始化张量统计
    LOG_DEBUG("启用调试模式: 张量统计");
#endif
#if DEBUG_MODE == 3
    runtime_manager_->setMode(MNN::Interpreter::Session_Debug);  // 启用调试模式
    _initDebug();                                                // 初始化调试信息
    LOG_DEBUG("启用调试模式: 完整调试");
#endif
    
    // 设置缓存文件路径
    {
        std::string cacheFilePath = tmpPath.length() != 0 ? tmpPath : ".";  // 默认使用当前目录
        runtime_manager_->setCache(cacheFilePath + "/mnn_cachefile.bin");   // 设置缓存文件
        LOG_DEBUG("设置缓存文件路径: " + cacheFilePath + "/mnn_cachefile.bin");
    }
    
    LOG_INFO("运行时环境初始化完成");
}

/**
 * @brief 加载模型权重和相关组件
 * 
 * 该方法用于加载模型的所有必要组件，包括运行时环境、分词器、磁盘嵌入和模型权重。
 * 同时会创建用于prefill和decode阶段的模块副本，以优化不同阶段的推理性能。
 */
void Llm::load() {
    LOG_INFO("开始加载模型");
    
    init_runtime();
    // init module status
    // 1. load vocab
    LOG_INFO("加载分词器");
    MNN_PRINT("load tokenizer\n");
    tokenizer_.reset(Tokenizer::createTokenizer(config_->tokenizer_file()));
    MNN_PRINT("load tokenizer Done\n");
    LOG_INFO("分词器加载完成");
    
    LOG_INFO("初始化磁盘嵌入");
    disk_embedding_.reset(new DiskEmbedding(config_));
    
    // 3. load model
    LOG_INFO("配置模块参数");
    Module::Config module_config;
    if (config_->backend_type() == "opencl" || config_->backend_type() == "vulkan") {
        module_config.shapeMutable = false;
        LOG_DEBUG("设置为固定形状模式(opencl/vulkan)");
    } else {
        module_config.shapeMutable = true;
        LOG_DEBUG("设置为可变形状模式");
    }
    module_config.rearrange    = true;
    // using base module for lora module
    if (base_module_ != nullptr) {
        module_config.base = base_module_;
        LOG_INFO("使用基础模块进行LoRA加载");
    }
    int layer_nums = config_->layer_nums();
    LOG_INFO("模型层数: " + std::to_string(layer_nums));
    
    // load single model
    modules_.resize(1);
    std::string model_path = config_->llm_model();
    LOG_INFO("加载模型文件: " + model_path);
    MNN_PRINT("load %s ... ", model_path.c_str());
    runtime_manager_->setExternalFile(config_->llm_weight());
    modules_[0].reset(Module::load(
                                       {"input_ids", "attention_mask", "position_ids"},
                                       {"logits"}, model_path.c_str(), runtime_manager_, &module_config));
    MNN_PRINT("Load Module Done!\n");
    LOG_INFO("主模块加载完成");
    
    LOG_INFO("创建解码模块副本");
    decode_modules_.resize(modules_.size());
    for (int v = 0; v < modules_.size(); ++v) {
        decode_modules_[v].reset(Module::clone(modules_[v].get()));
    }
    MNN_PRINT("Clone Decode Module Done!\n");
    LOG_INFO("解码模块副本创建完成");

    prefill_modules_ = modules_;
    LOG_INFO("模型加载完成");
}

/**
 * @brief 应用LoRA权重
 * 
 * 该方法用于加载并应用LoRA（Low-Rank Adaptation）权重到模型中。
 * LoRA是一种高效的微调技术，通过低秩矩阵分解来减少参数量。
 * 
 * @param lora_path LoRA权重文件路径
 * @return LoRA模块在模块列表中的索引
 */
size_t Llm::apply_lora(const std::string& lora_path) {
    std::string model_path = config_->base_dir_ + "/" + lora_path;
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange    = true;
    module_config.base         = modules_.begin()->get();
    size_t lora_index          = modules_.size();
    modules_.emplace_back(Module::load({"input_ids", "attention_mask", "position_ids"}, {"logits"},
                                           model_path.c_str(), runtime_manager_, &module_config));
    select_module(lora_index);
    return lora_index;
}

/**
 * @brief 创建LoRA模型实例
 * 
 * 该方法用于创建一个新的LoRA模型实例，该实例基于当前模型的配置，
 * 但使用指定的LoRA权重文件。
 * 
 * @param lora_path LoRA权重文件路径
 * @return 新创建的LoRA模型实例指针
 */
Llm* Llm::create_lora(const std::string& lora_path) {
    auto llm = new Llm(config_);
    llm->set_config("{\"llm_model\": \"" + lora_path + "\"}");
    llm->base_module_ = modules_.begin()->get();
    llm->load();
    return llm;
}

/**
 * @brief 释放指定索引的模块
 * 
 * 该方法用于释放指定索引的模型模块，释放其占用的资源。
 * 如果要释放的模块是当前prefill模块，则会切换回默认模块。
 * 
 * @param index 要释放的模块索引
 * @return 释放成功返回true，否则返回false
 */
bool Llm::release_module(size_t index) {
    if (index >= modules_.size()) {
        return false;
    }
    if (prefill_modules_[0] == modules_[index]) {
        select_module(0);
    }
    modules_[index].reset();
    return true;
}

/**
 * @brief 选择指定索引的模块作为当前模块
 * 
 * 该方法用于选择指定索引的模型模块作为当前使用的模块，
 * 并更新prefill和decode阶段使用的模块。
 * 
 * @param index 要选择的模块索引
 * @return 选择成功返回true，否则返回false
 */
bool Llm::select_module(size_t index) {
    if (index >= modules_.size()) {
        return false;
    }
    if (modules_[index] == nullptr) {
        return false;
    }
    if (decode_modules_.empty()) {
        decode_modules_.resize(modules_.size());
        prefill_modules_.resize(modules_.size());
    }
    decode_modules_[0].reset(Module::clone(modules_[index].get()));
    prefill_modules_[0] = modules_[index];
    return true;
}

/**
 * @brief 开启或关闭性能追踪
 * 
 * 该方法用于开启或关闭模型推理的性能追踪功能。
 * 当开启时，会收集模型推理过程中的性能数据用于分析和优化。
 * 
 * @param start true表示开启追踪，false表示关闭追踪
 */
void Llm::trace(bool start) {
    auto status = MNN::Interpreter::Session_Resize_Check;
    if (start) {
        status = MNN::Interpreter::Session_Resize_Check;
    } else {
        status = MNN::Interpreter::Session_Resize_Fix;
    }
    for (auto& m : decode_modules_) {
        m->traceOrOptimize(status);
    }

    runtime_manager_->updateCache();
    mTracing = start;
}

/**
 * @brief 微调模型参数以优化性能
 * 
 * 该方法用于微调模型的特定参数以优化推理性能。
 * 目前主要支持优化Metal后端的OP编码器数量参数。
 * 
 * @param type 微调参数类型
 * @param candidates 候选参数值列表
 */
void Llm::tuning(TuneType type, std::vector<int> candidates) {
    if (type != OP_ENCODER_NUMBER) {
        MNN_ERROR("tuning type not supported\n");
        return;
    }
    if (config_->backend_type() != "metal") {
        return;
    }

    current_modules_     = decode_modules_;
    int64_t min_time     = INT64_MAX;
    int prefer_candidate = 10;
    for (auto& candidate : candidates) {
        runtime_manager_->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, candidate);

        auto st     = std::chrono::system_clock::now();
        auto logits = forward({0});
        if (nullptr == logits.get()) {
            return;
        }
        if (logits->getInfo()->size == 0) {
            return;
        }
        auto token   = sample(logits, {});
        auto et      = std::chrono::system_clock::now();
        int64_t time = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
        if (time < min_time) {
            prefer_candidate = candidate;
            min_time         = time;
            // MNN_PRINT("op encode number:%d, decode time: %lld us\n", candidate, time);
        }
    }
    runtime_manager_->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, prefer_candidate);
}
/**
 * @brief 切换模型推理阶段
 * 
 * 该方法用于在Prefill阶段和Decode阶段之间切换模型模块。
 * Prefill阶段处理完整的输入序列，Decode阶段逐个生成输出token。
 * 
 * @param stage 目标推理阶段（Prefill或Decode）
 */
void Llm::switchMode(Llm::Stage stage) {
    switch (stage) {
        case Prefill:
            current_modules_ = prefill_modules_;
            break;
        case Decode:
            current_modules_ = decode_modules_;
            break;
        default:
            break;
    }
}

/**
 * @brief 原始前向推理接口
 * 
 * 该方法是模型前向推理的核心接口，接收隐藏状态、注意力掩码和位置信息作为输入，
 * 通过当前模块进行推理计算，返回logits输出。
 * 
 * @param hiddenState 隐藏状态输入张量
 * @param mask 注意力掩码张量
 * @param inputPos 输入位置信息张量
 * @return 推理结果logits张量，推理失败返回nullptr
 */
MNN::Express::VARP Llm::forwardRaw(MNN::Express::VARP hiddenState, MNN::Express::VARP mask, MNN::Express::VARP inputPos) {

    LOG_DEBUG("forwardRaw start");
    // 声明用于存储输出logits的变量
    VARP logits;
    
    // 创建输出向量容器，用于接收模型的前向推理结果
    std::vector<MNN::Express::VARP> outputs;
    
    // 调用当前模块的前向推理方法，传入隐藏状态、注意力掩码和位置信息
    // current_modules_.back() 获取当前使用的模块（prefill或decode模块）
    // onForward 执行实际的神经网络前向计算过程
    outputs = current_modules_.back()->onForward({hiddenState, mask, inputPos});
    
    // 打印outputs的详细信息用于调试和监控
    std::string outputs_info = "前向推理完成，outputs数量: " + std::to_string(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        outputs_info += " | outputs[" + std::to_string(i) + "]";
        if (outputs[i].get() != nullptr) {
            auto info = outputs[i]->getInfo();
            std::string dims_str = "[";
            for (size_t j = 0; j < info->dim.size(); ++j) {
                if (j > 0) dims_str += ", ";
                dims_str += std::to_string(info->dim[j]);
            }
            dims_str += "]";
            outputs_info += ": 维度" + dims_str + ", 总元素数" + std::to_string(info->size) + 
                           ", 数据类型" + (info->type == halide_type_of<float>() ? "float" : "other");
        } else {
            outputs_info += ": nullptr";
        }
    }
    LOG_DEBUG(outputs_info);

    // 检查输出是否为空，如果为空则说明推理失败
    if (outputs.empty()) {
        LOG_ERROR("前向推理失败：outputs为空");
        return nullptr;
    }
    
    // 提取第一个输出作为logits（通常模型只有一个输出）
    // logits包含了词汇表中每个token的未归一化概率分数
    logits = outputs[0];

    LOG_DEBUG("forwardRaw end");
    // 返回logits张量，供后续的采样或其他处理使用
    return logits;
}

/**
 * @brief 前向推理接口
 * 
 * 该方法是模型前向推理的高级接口，接收token ID序列作为输入，
 * 自动处理嵌入生成、注意力掩码和位置ID生成等步骤，
 * 调用原始前向推理接口完成推理计算。
 * 
 * @param input_ids 输入的token ID序列
 * @return 推理结果logits张量，推理失败返回nullptr
 */
VARP Llm::forward(const std::vector<int>& input_ids) {
    // 获取输入序列长度
    int seq_len         = input_ids.size();
    
    // 生成注意力掩码，用于控制模型在计算注意力时哪些位置可以被关注
    auto attention_mask = gen_attention_mask(seq_len);
    
    // 生成位置编码ID，用于表示序列中每个token的位置信息
    auto position_ids = gen_position_ids(seq_len);
    
    // 将输入的token ID序列转换为嵌入向量表示
    auto hidden_states = embedding(input_ids);
    
    // 调用原始前向推理函数，传入嵌入向量、注意力掩码和位置ID
    // 执行Transformer模型的完整前向计算过程
    auto logits = forwardRaw(hidden_states, attention_mask, position_ids);
    
    // 更新状态：累加总序列长度
    mState.all_seq_len_ += seq_len;
    
    // 更新状态：递增生成序列计数器
    mState.gen_seq_len_++;
    
    // 返回模型输出的logits（未归一化的概率分布）
    return logits;
}

/**
 * @brief 采样方法：从logits中选择下一个token
 * 
 * 该方法用于从模型输出的logits中采样下一个token。
 * 使用贪心策略（argmax）并应用重复惩罚来提高生成质量，
 * 避免生成重复的内容。
 * 
 * @param logits 模型输出的logits张量
 * @param pre_ids 历史生成的token ID序列
 * @param offset 采样起始偏移位置
 * @param size 采样范围大小
 * @return 采样得到的token ID
 */
int Llm::sample(VARP logits, const std::vector<int>& pre_ids, int offset, int size) {
    std::unordered_set<int> ids_set(pre_ids.begin(), pre_ids.end());  // 将历史token转为集合，用于快速查找
    auto scores = (float*)(logits->readMap<float>()) + offset;        // 获取logits分数指针
    
    if (0 == size) {
        size = logits->getInfo()->size;  // 如果未指定大小，使用logits的完整大小
    }
    
    // 应用重复惩罚：降低已生成token的概率，避免重复
    const float repetition_penalty = 1.1;  // 重复惩罚系数
    for (auto id : ids_set) {
        float score = scores[id];
        // 如果分数为负，增大惩罚；如果为正，减小奖励
        scores[id] = score < 0 ? score * repetition_penalty : score / repetition_penalty;
    }
    
    // 贪心搜索：选择分数最高的token（argmax）
    float max_score = scores[0];  // 初始化最大分数
    int token_id = 0;             // 初始化选中的token ID
    
    for (int i = 1; i < size; i++) {
        float score = scores[i];
        if (score > max_score) {
            max_score = score;  // 更新最大分数
            token_id  = i;      // 更新最佳token ID
        }
    }
    
    return token_id;  // 返回选中的token ID
}

/**
 * @brief 应用模板生成提示词
 * 
 * 该函数用于将内容和角色信息应用到提示词模板中，
 * 生成符合模型要求的输入格式。
 * 
 * @param prompt_template 提示词模板字符串
 * @param content 要插入模板的内容
 * @param role 角色信息（可选）
 * @return 应用模板后生成的字符串
 */
static std::string apply_template(std::string prompt_template, const std::string& content,
                                  const std::string& role = "") {
    if (prompt_template.empty())
        return content;
    if (!role.empty()) {
        const std::string placeholder = "%r";
        size_t start_pos              = prompt_template.find(placeholder);
        if (start_pos == std::string::npos)
            return content;
        prompt_template.replace(start_pos, placeholder.length(), role);
    }
    const std::string placeholder = "%s";
    size_t start_pos              = prompt_template.find(placeholder);
    if (start_pos == std::string::npos)
        return content;
    prompt_template.replace(start_pos, placeholder.length(), content);
    return prompt_template;
}

/**
 * @brief 应用提示词模板
 * 
 * 该方法用于将用户输入内容应用到配置的提示词模板中，
 * 生成符合模型要求的输入格式。
 * 
 * @param user_content 用户输入的内容
 * @return 应用模板后生成的提示词字符串
 */
std::string Llm::apply_prompt_template(const std::string& user_content) const {
    auto chat_prompt = config_->prompt_template();
    return apply_template(chat_prompt, user_content);
}

/**
 * @brief 应用聊天模板
 * 
 * 该方法用于将聊天历史记录应用到配置的聊天模板中，
 * 生成符合模型要求的多轮对话输入格式。
 * 
 * @param chat_prompts 聊天历史记录，包含角色和内容的pair序列
 * @return 应用模板后生成的聊天提示词字符串
 */
std::string Llm::apply_chat_template(const std::vector<PromptItem>& chat_prompts) const {
    auto chat_template = config_->chat_template();
    std::string prompt_result;
    auto iter = chat_prompts.begin();
    for (; iter != chat_prompts.end() - 1; ++iter) {
        prompt_result += apply_template(chat_template, iter->second, iter->first);
    }
    if (iter->first == "user") {
        prompt_result += apply_prompt_template(iter->second);
    } else {
        prompt_result += apply_template(chat_template, iter->second, iter->first);
    }
    return prompt_result;
}

/**
 * @brief 启动聊天交互
 * 
 * 该方法用于启动一个交互式的聊天会话，用户可以通过命令行与模型进行多轮对话。
 * 支持 /exit 命令退出聊天，/reset 命令重置对话历史。
 */
void Llm::chat() {
    std::vector<PromptItem> history;
    history.push_back(std::make_pair("system", "You are a helpful assistant."));
    while (true) {
        std::cout << "\nQ: ";
        std::string user_str;
        std::cin >> user_str;
        if (user_str == "/exit") {
            break;
        }
        if (user_str == "/reset") {
            history.resize(1);
            std::cout << "\nA: reset done." << std::endl;
            continue;
        }
        std::cout << "\nA: " << std::flush;
        if (config_->reuse_kv()) {
            response(user_str);
        } else {
            history.emplace_back(std::make_pair("user", user_str));
            std::ostringstream lineOs;
            response(history, &lineOs, nullptr, 1);
            auto line = lineOs.str();
            while (!stoped() && mState.gen_seq_len_ < config_->max_new_tokens()) {
                std::cout << line << std::flush;
                lineOs.str("");
                generate(1);
                line = lineOs.str();
            }
            history.emplace_back(std::make_pair("assistant", line));
        }
        std::cout << std::endl;
    }
}

/**
 * @brief 重置模型状态
 * 
 * 该方法用于重置模型的内部状态，清空历史记录和序列长度计数器，
 * 使模型回到初始状态。
 */
void Llm::reset() {
    mState.history_ids_.clear();
    mState.all_seq_len_ = 0;
}

/**
 * @brief 初始化生成过程
 * 
 * 该方法用于初始化文本生成过程的各种状态变量，
 * 包括输出流、计数器、计时器等，并根据配置决定是否重用KV缓存。
 * 
 * @param os 输出流指针，默认为nullptr
 * @param end_with 结束标记字符串，默认为nullptr
 */
void Llm::generate_init(std::ostream* os, const char* end_with) {
    // init status
    mState.os_ = os;
    if (nullptr != end_with) {
        mState.end_with_ = end_with;
    }
    mState.gen_seq_len_ = 0;
    mState.vision_us_   = 0;
    mState.audio_us_    = 0;
    mState.prefill_us_  = 0;
    mState.decode_us_   = 0;
    mState.current_token_ = 0;
    if (!config_->reuse_kv()) {
        mState.all_seq_len_ = 0;
        mState.history_ids_.clear();
        mMeta->remove = mMeta->previous;
    }
    mState.output_ids_.clear();
    current_modules_ = prefill_modules_;
}
/**
 * @brief 获取当前历史记录长度
 * 
 * 该方法用于获取当前KV缓存中保存的历史记录长度。
 * 
 * @return 当前历史记录长度
 */
size_t Llm::getCurrentHistory() const {
    return mMeta->previous;
}

/**
 * @brief 删除指定范围的历史记录
 * 
 * 该方法用于删除KV缓存中指定范围的历史记录，
 * 以控制缓存大小和处理长序列。
 * 
 * @param begin 删除起始位置
 * @param end 删除结束位置，默认为0表示删除到末尾
 */
void Llm::eraseHistory(size_t begin, size_t end) {
    if (0 == end) {
        end = mMeta->previous;
    }
    if (end > mMeta->previous || begin >= end) {
        MNN_ERROR("Invalid erase range history larger than current\n");
        return;
    }
    if (mMeta->remove != 0) {
        MNN_ERROR("MNN-LLM: erase history hasn't been executed by response, override erase info\n");
    }
    mMeta->remove = mMeta->previous - begin;
    if (end != mMeta->previous) {
        mMeta->reserveHost.resize(2);
        mMeta->reserve = mMeta->reserveHost.data();
        mMeta->n_reserve = 1;
        mMeta->reserve[0] = end - begin;
        mMeta->reserve[1] = mMeta->previous - end;
    }
}

/**
 * @brief 检查是否应该停止生成
 * 
 * 该方法用于检查当前生成的token是否为停止token，
 * 如果是则返回true，表示应该停止生成。
 * 
 * @return 如果应该停止生成返回true，否则返回false
 */
bool Llm::stoped() {
    return is_stop(mState.current_token_);
}

/**
 * @brief 增量生成方法：逐个生成token直到达到最大数量或遇到停止条件
 * 
 * 该方法是解码阶段的核心循环，每次生成一个新token，直到达到指定的最大token数量
 * 或遇到停止条件。在生成过程中会实时输出结果并更新内部状态。
 * 
 * @param max_token 最大生成token数量
 */
void Llm::generate(int max_token) {
    int len = 0;  // 已生成的token数量计数器
    
    while (len < max_token) {
        auto st = std::chrono::system_clock::now();  // 记录开始时间，用于性能统计
        
        // 如果设置了输出流，实时输出解码的token
        if (nullptr != mState.os_) {
            *mState.os_ << tokenizer_decode(mState.current_token_);  // 解码并输出当前token
            *mState.os_ << std::flush;                               // 立即刷新输出
        }
        
        // 更新状态和KV缓存信息
        mState.history_ids_.push_back(mState.current_token_);  // 将当前token添加到历史记录
        mMeta->add = 1;      // 标记新增1个token到KV缓存
        mMeta->remove = 0;   // 无需移除任何token
        
        // 前向推理：使用当前token生成下一个token的logits
        auto logits = forward({mState.current_token_});
        mMeta->sync();       // 同步KV缓存状态
        len++;              // 增加生成计数
        
        // 检查推理是否成功
        if (nullptr == logits.get()) {
            break;  // 推理失败，退出
        }
        if (logits->getInfo()->size == 0) {
            break;  // logits为空，退出
        }
        
        // 采样下一个token
        mState.current_token_ = sample(logits, mState.history_ids_);
        
        auto et = std::chrono::system_clock::now();  // 记录结束时间
        // 累计解码时间统计
        mState.decode_us_ += std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
        
        // 检查是否遇到停止token
        if (is_stop(mState.current_token_) && nullptr != mState.os_) {

            *mState.os_ << mState.end_with_ << std::flush;  // 输出结束标记
            break;  // 遇到停止token，结束生成
        }
    }
}

/**
 * @brief 完整生成方法：给定输入token序列，生成指定数量的新token
 * 
 * 该方法是LLM文本生成的主要入口，包含预填充（prefill）和解码（decode）两个阶段。
 * 预填充阶段处理整个输入序列，解码阶段逐个生成新token。
 * 
 * @param input_ids 输入的token ID序列
 * @param max_tokens 最大生成token数量，负数表示使用配置默认值
 * @return 生成的token ID序列
 */
std::vector<int> Llm::generate(const std::vector<int>& input_ids, int max_tokens) {
    LOG_INFO("开始生成过程，输入token数量: " + std::to_string(input_ids.size()));
    
    if (max_tokens < 0) {
        max_tokens = config_->max_new_tokens();  // 使用配置文件中的默认最大token数
    }
    
    LOG_INFO("最大生成token数量: " + std::to_string(max_tokens));
    
    // 设置KV缓存和状态信息
    mMeta->add = input_ids.size();  // 标记需要添加的token数量（即输入序列长度）
    mState.prompt_len_ = static_cast<int>(input_ids.size());  // 记录提示词长度
    // 将输入token序列添加到历史记录中
    mState.history_ids_.insert(mState.history_ids_.end(), input_ids.begin(), input_ids.end());
    
    auto st = std::chrono::system_clock::now();  // 记录预填充开始时间
    
    // === 预填充阶段（Prefill Phase）===
    LOG_INFO("开始预填充阶段");
    // 使用预填充模块处理整个输入序列，一次性计算所有输入token的KV缓存
    current_modules_ = prefill_modules_;
    auto logits = forward(input_ids);  // 前向推理，生成最后一个位置的logits
    
    if (nullptr == logits.get()) {
        LOG_ERROR("预填充阶段失败，logits为空");
        return {};  // 预填充失败，返回空结果
    }
    
    // 采样第一个生成的token
    mState.current_token_ = sample(logits, mState.history_ids_);
    LOG_DEBUG("预填充完成，首个生成token: " + std::to_string(mState.current_token_));
    logits = nullptr;  // 释放logits内存
    
    auto et = std::chrono::system_clock::now();  // 记录预填充结束时间
    
    // === 解码阶段准备（Decode Phase Setup）===
    LOG_INFO("切换到解码阶段");
    current_modules_ = decode_modules_;  // 切换到解码模块（针对单token推理优化）
    // 记录预填充阶段耗时
    mState.prefill_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    LOG_DEBUG("预填充耗时: " + std::to_string(mState.prefill_us_ / 1000.0) + " ms");
    mMeta->sync();  // 同步KV缓存状态
    
    // === 解码阶段（Decode Phase）===
    LOG_INFO("开始解码阶段，目标生成token数: " + std::to_string(max_tokens));
    // 逐个生成新token，直到达到最大数量或遇到停止条件
    generate(max_tokens);

#ifdef DUMP_PROFILE_INFO
    print_speed();  // 如果启用性能分析，打印速度统计信息
#endif
    
    LOG_INFO("生成过程完成，实际生成token数: " + std::to_string(mState.gen_seq_len_));
    return mState.output_ids_;  // 返回生成的token序列
}

/**
 * @brief 将文本编码为token ID序列
 * 
 * 该方法用于将用户输入的文本内容编码为token ID序列，
 * 可选择是否应用提示词模板。
 * 
 * @param user_content 用户输入的文本内容
 * @param use_template 是否应用提示词模板，默认为true
 * @return 编码后的token ID序列
 */
std::vector<int> Llm::tokenizer_encode(const std::string& user_content, bool use_template) {
    if (!use_template) {
        return tokenizer_->encode(user_content);
    }
    auto prompt    = apply_prompt_template(user_content);
    auto input_ids = tokenizer_->encode(prompt);
    return input_ids;
}

/**
 * @brief 对用户输入进行响应
 * 
 * 该方法用于对用户输入的文本内容进行响应，生成相应的回复文本。
 * 会自动处理文本编码、模型推理和文本生成等过程。
 * 
 * @param user_content 用户输入的文本内容
 * @param os 输出流指针，默认为标准输出
 * @param end_with 结束标记字符串，默认为换行符
 * @param max_new_tokens 最大生成token数量，默认为-1表示使用配置默认值
 */
void Llm::response(const std::string& user_content, std::ostream* os, const char* end_with, int max_new_tokens) {
    if (!end_with) {
        end_with = "\n";
    }
    generate_init(os, end_with);
    std::vector<int> input_ids;
    input_ids = tokenizer_encode(user_content);
    
    // 构建input_ids的完整日志信息
    std::string ids_str = "input_ids: [";
    for (size_t i = 0; i < input_ids.size(); ++i) {
        if (i > 0) ids_str += ", ";
        ids_str += std::to_string(input_ids[i]);
    }
    ids_str += "]";
    LOG_INFO(ids_str);

    generate(input_ids, max_new_tokens);
}

/**
 * @brief 对聊天历史进行响应
 * 
 * 该方法用于对聊天历史记录进行响应，生成相应的回复文本。
 * 会自动应用聊天模板并调用单次响应方法。
 * 
 * @param chat_prompts 聊天历史记录
 * @param os 输出流指针，默认为标准输出
 * @param end_with 结束标记字符串，默认为nullptr
 * @param max_new_tokens 最大生成token数量，默认为-1表示使用配置默认值
 */
void Llm::response(const std::vector<PromptItem>& chat_prompts, std::ostream* os, const char* end_with, int max_new_tokens) {
    if (chat_prompts.empty()) {
        return;
    }
    auto prompt = apply_chat_template(chat_prompts);
    response(prompt, os, end_with, max_new_tokens);
}

/**
 * @brief Llm类构造函数
 * 
 * 该构造函数用于初始化Llm对象，设置模型配置并创建KV缓存元数据。
 * 
 * @param config 模型配置指针
 */
Llm::Llm(std::shared_ptr<LlmConfig> config) : config_(config) {
    // 初始化日志系统
    Logger::getInstance().init();
    LOG_INFO("创建Llm实例");
    
    mMeta.reset(new KVMeta);
    LOG_DEBUG("KV缓存元数据初始化完成");
}

/**
 * @brief Llm类析构函数
 * 
 * 该析构函数用于清理Llm对象占用的资源，包括模块、运行时管理器等。
 * 如果启用了调试模式，还会打印性能统计信息。
 */
Llm::~Llm() {
    LOG_INFO("开始销毁Llm实例");
    
#if DEBUG_MODE == 1
    if (nullptr != gTimeTraceInfo) {
        float opSummer       = 0.0f;
        float opFlopsSummber = 0.0f;
        for (auto& iter : gTimeTraceInfo->mTypes) {
            float summer      = 0.0f;
            float summerflops = 0.0f;
            for (auto& t : iter.second) {
                for (auto& t0 : t.second) {
                    summer += t0.first;
                    summerflops += t0.second;
                }
            }
            summer      = summer;
            summerflops = summerflops;
            MNN_PRINT("%s : %.7f, FLOP: %.7f, Speed: %.7f GFlops\n", iter.first.c_str(), summer, summerflops,
                      summerflops / summer);
            opSummer += summer;
            opFlopsSummber += summerflops;
        }
        MNN_PRINT("OP Summer: %.7f, Flops: %.7f, Speed: %.7f GFlops\n", opSummer, opFlopsSummber,
                  opFlopsSummber / opSummer);
    }
#endif
    
    LOG_DEBUG("清理模块资源");
    current_modules_.clear();
    decode_modules_.clear();
    prefill_modules_.clear();
    modules_.clear();
    runtime_manager_.reset();
    
    LOG_INFO("Llm实例销毁完成");
    Logger::getInstance().flush();
}

/**
 * @brief 打印推理速度统计信息
 * 
 * 该方法用于打印模型推理过程中的各种性能统计数据，
 * 包括各阶段耗时、处理速度等信息。
 */
void Llm::print_speed() {
    auto vision_s   = mState.vision_us_ * 1e-6;
    auto audio_s   = mState.audio_us_ * 1e-6;
    auto prefill_s = mState.prefill_us_ * 1e-6;
    auto decode_s  = mState.decode_us_ * 1e-6;
    auto total_s   = vision_s + audio_s + prefill_s + decode_s;
    printf("\n#################################\n");
    printf(" total tokens num  = %d\n", mState.prompt_len_ + mState.gen_seq_len_);
    printf("prompt tokens num  = %d\n", mState.prompt_len_);
    printf("output tokens num  = %d\n", mState.gen_seq_len_);
    printf("  total time = %.2f s\n", total_s);
    if (1 || vision_s) {
    printf(" vision time = %.2f s\n", audio_s);
    }
    if (1 || audio_s) {
    printf("  audio time = %.2f s\n", audio_s);
    }
    printf("prefill time = %.2f s\n", prefill_s);
    printf(" decode time = %.2f s\n", decode_s);
    printf("  total speed = %.2f tok/s\n", (mState.prompt_len_ + mState.gen_seq_len_) / total_s);
    printf("prefill speed = %.2f tok/s\n", mState.prompt_len_ / prefill_s);
    printf(" decode speed = %.2f tok/s\n", mState.gen_seq_len_ / decode_s);
    printf("   chat speed = %.2f tok/s\n", mState.gen_seq_len_ / total_s);
    printf("##################################\n");
}

/**
 * @brief 判断是否需要创建新的变量张量
 * 
 * 该函数用于判断是否需要创建新的变量张量，根据张量是否为空或维度是否匹配来决定。
 * 
 * @param var 变量张量指针
 * @param axis 需要检查的维度索引
 * @param seq_len 序列长度
 * @return 如果需要创建新变量返回true，否则返回false
 */
static inline bool needNewVar(VARP var, int axis, int seq_len) {
    if (var == nullptr) {
        return true;
    }
    if (var->getInfo()->dim[axis] != seq_len) {
        return true;
    }
    return false;
}

/**
 * @brief 根据输入的token ID序列生成对应的嵌入向量
 * 
 * 该方法用于将输入的token ID序列转换为嵌入向量，使用磁盘嵌入技术以节省内存。
 * 
 * @param input_ids 输入的token ID序列
 * @return 生成的嵌入向量张量
 */
VARP Llm::embedding(const std::vector<int>& input_ids) {
    AUTOTIME;
    int hidden_size = config_->hidden_size();
    int seq_len = static_cast<int>(input_ids.size());
    VARP res = _Input({seq_len, 1, hidden_size}, NCHW);
    // disk embedding to save memory
    disk_embedding_->embedding(input_ids, res->writeMap<float>());
    return res;
}

/**
 * @brief 将token ID解码为对应的文本
 * 
 * 该方法用于将token ID解码为对应的文本，并处理UTF-8编码的特殊情况。
 * 
 * @param id 要解码的token ID
 * @return 解码后的文本
 */
std::string Llm::tokenizer_decode(int id) {
    std::string word = tokenizer_->decode(id);
    // Fix utf-8 garbled characters
    if (word.length() == 6 && word[0] == '<' && word[word.length() - 1] == '>' && word[1] == '0' && word[2] == 'x') {
        int num = std::stoi(word.substr(3, 2), nullptr, 16);
        word    = static_cast<char>(num);
    }
    return word;
}

/**
 * @brief 生成注意力掩码
 * 
 * 该方法用于生成注意力机制所需的掩码张量，支持多种掩码类型和数据格式。
 * 
 * @param seq_len 序列长度
 * @return 生成的注意力掩码张量
 */
VARP Llm::gen_attention_mask(int seq_len) {
    int kv_seq_len = mState.all_seq_len_ + seq_len;
    if (seq_len == 1) {
        kv_seq_len = seq_len;
    }
    if (config_->attention_mask() == "float") {
        if (needNewVar(attention_mask_, 2, seq_len)) {
            attention_mask_ = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<float>());
        } else {
            return attention_mask_;
        }
        auto ptr = attention_mask_->writeMap<float>();
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_seq_len; j++) {
                int row = i + mState.all_seq_len_;
                ptr[kv_seq_len * i + j] = (j > row) * std::numeric_limits<float>::lowest();
            }
        }
        return attention_mask_;
    } else {
        if (needNewVar(attention_mask_, 2, seq_len)) {
            attention_mask_ = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<int>());
        } else {
            return attention_mask_;
        }
        auto ptr = attention_mask_->writeMap<int>();
        if (config_->attention_mask() == "glm") {
            // chatglm
            for (int i = 0; i < seq_len * kv_seq_len; i++) {
                ptr[i] = 0;
            }
            if (seq_len > 1) {
                for (int i = 1; i < seq_len; i++) {
                    ptr[seq_len * i - 1] = 1;
                }
            }
        } else {
            bool is_glm2 = config_->attention_mask() == "glm2";
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < kv_seq_len; j++) {
                    int row              = i + mState.all_seq_len_;
                    ptr[seq_len * i + j] = is_glm2 ? j > row : j <= row;
                }
            }
        }
        return attention_mask_;
    }
}

/**
 * @brief 生成位置ID
 * 
 * 该方法用于生成位置编码所需的位置ID张量，支持多种位置编码格式。
 * 
 * @param seq_len 序列长度
 * @return 生成的位置ID张量
 */
VARP Llm::gen_position_ids(int seq_len) {
    if (config_->attention_mask() == "glm") {
        // chatglm
        if (needNewVar(position_ids_, 2, seq_len)) {
            position_ids_ = _Input({1, 2, seq_len}, NCHW, halide_type_of<int>());
        }
        auto ptr = position_ids_->writeMap<int>();
        if (seq_len == 1) {
            ptr[0] = mState.all_seq_len_ - mState.gen_seq_len_ - 2;
            ptr[1] = mState.gen_seq_len_ + 1;
        } else {
            for (int i = 0; i < seq_len - 1; i++) {
                ptr[i]           = i;
                ptr[seq_len + i] = 0;
            }
            ptr[seq_len - 1]     = seq_len - 2;
            ptr[2 * seq_len - 1] = 1;
        }
        return position_ids_;
    } else {
        bool is_glm2 = config_->attention_mask() == "glm2";
        if (needNewVar(position_ids_, 0, seq_len)) {
            position_ids_ = _Input({seq_len}, NCHW, halide_type_of<int>());
        }
        auto ptr = position_ids_->writeMap<int>();
        if (seq_len == 1) {
            ptr[0] = is_glm2 ? mState.gen_seq_len_ : mState.all_seq_len_;
        } else {
            for (int i = 0; i < seq_len; i++) {
                ptr[i] = i + mState.all_seq_len_;
            }
        }
        return position_ids_;
    }
}

/**
 * @brief 判断token ID是否为停止符
 * 
 * 该方法用于判断指定的token ID是否为模型定义的停止符。
 * 
 * @param token_id 要判断的token ID
 * @return 如果是停止符返回true，否则返回false
 */
bool Llm::is_stop(int token_id) {
    return tokenizer_->is_stop(token_id);
}

/**
 * @brief 加载多模态模型
 * 
 * 该方法用于加载多模态模型，包括基础的LLM模型和多模态处理模块（视觉或音频）。
 * 会根据配置初始化运行时环境和相应的处理模块。
 */
void Mllm::load() {
    Llm::load();
    if (config_->mllm_config_.empty()) {
        mllm_runtime_manager_ = runtime_manager_;
    } else {
        ScheduleConfig config;
        BackendConfig cpuBackendConfig;
        config.type      = backend_type_convert(config_->backend_type(true));;
        config.numThread = config_->thread_num(true);
        if (config_->power(true) == "high") {
            cpuBackendConfig.power = BackendConfig::Power_High;
        } else if (config_->power(true) == "low") {
            cpuBackendConfig.power = BackendConfig::Power_Low;
        }
        if (config_->memory(true) == "high") {
            cpuBackendConfig.memory = BackendConfig::Memory_High;
        } else if (config_->memory(true) == "low") {
            cpuBackendConfig.memory = BackendConfig::Memory_Low;
        }
        if (config_->precision(true) == "high") {
            cpuBackendConfig.precision = BackendConfig::Precision_High;
        } else if (config_->precision(true) == "low") {
            cpuBackendConfig.precision = BackendConfig::Precision_Low;
        }
        config.backendConfig = &cpuBackendConfig;
        mllm_runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
        mllm_runtime_manager_->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
        mllm_runtime_manager_->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, 1); // 1: per batch quant, 2: per tensor quant
        mllm_runtime_manager_->setHint(MNN::Interpreter::QKV_QUANT_OPTIONS, config_->quant_qkv());
        mllm_runtime_manager_->setHint(MNN::Interpreter::KVCACHE_SIZE_LIMIT, config_->kvcache_limit());
        std::string tmpPath = config_->tmp_path();
        if (config_->kvcache_mmap()) {
            mllm_runtime_manager_->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
        }
        if (config_->use_mmap()) {
            mllm_runtime_manager_->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
        }
    }
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange    = true;
    if (config_->is_visual()) {
        mllm_runtime_manager_->setExternalFile(config_->visual_model() + ".weight");
        mul_module_.reset(Module::load({}, {}, config_->visual_model().c_str(), mllm_runtime_manager_, &module_config));
    }
    if (config_->is_audio()) {
        mllm_runtime_manager_->setExternalFile(config_->audio_model() + ".weight");
        mul_module_.reset(Module::load({}, {}, config_->audio_model().c_str(), mllm_runtime_manager_, &module_config));
    }
}

static void dump_impl(const float *signal, size_t size, int row = 0) {
if (row) {
int col = size / row;
printf("# %d, %d: [\n", row, col);
for (int i = 0; i < 3; i++) {
for (int j = 0; j < 3; j++) {
printf("%f, ", signal[i * col + j]);
}
printf("..., ");
for (int j = col - 3; j < col; j++) {
printf("%f, ", signal[i * col + j]);
}
printf("\n");
}
printf("..., \n");
for (int i = row - 3; i < row; i++) {
for (int j = 0; j < 3; j++) {
printf("%f, ", signal[i * col + j]);
}
printf("..., ");
for (int j = col - 3; j < col; j++) {
printf("%f, ", signal[i * col + j]);
}
printf("\n");
}
printf("]\n");
} else {
printf("# %lu: [", size);
for (int i = 0; i < 3; i++) {
printf("%f, ", signal[i]);
}
printf("..., ");
for (int i = size - 3; i < size; i++) {
printf("%f, ", signal[i]);
}
printf("]\n");
}
}

void dump_var(VARP var) {
auto dims    = var->getInfo()->dim;
bool isfloat = true;
printf("{\ndtype = ");
if (var->getInfo()->type == halide_type_of<float>()) {
printf("float");
isfloat = true;
} else if (var->getInfo()->type == halide_type_of<int>()) {
printf("int");
isfloat = false;
}
printf("\nformat = %d\n", var->getInfo()->order);
printf("\ndims = [");
for (int i = 0; i < dims.size(); i++) {
printf("%d ", dims[i]);
}
printf("]\n");

if (isfloat) {
if ((dims.size() > 2 && dims[1] > 1 && dims[2] > 1) || (dims.size() == 2 && dims[0] > 1 && dims[1] > 1)) {
int row = dims[dims.size() - 2];
dump_impl(var->readMap<float>(), var->getInfo()->size, row);
} else {
printf("data = [");
auto total = var->getInfo()->size;
if (total > 32) {
for (int i = 0; i < 5; i++) {
printf("%f ", var->readMap<float>()[i]);
}
printf("..., ");
for (int i = total - 5; i < total; i++) {
printf("%f ", var->readMap<float>()[i]);
}
} else {
for (int i = 0; i < total; i++) {
printf("%f ", var->readMap<float>()[i]);
}
}
printf("]\n}\n");
}
} else {
printf("data = [");
int size = var->getInfo()->size > 10 ? 10 : var->getInfo()->size;
for (int i = 0; i < size; i++) {
printf("%d ", var->readMap<int>()[i]);
}
printf("]\n}\n");
}
}

// 视觉处理方法：将图像文件转换为token序列
// 支持多种视觉模型架构，包括标准模型和Qwen2-VL等
std::vector<int> Mllm::vision_process(const std::string& file) {
#ifdef LLM_SUPPORT_VISION
    VARP image = MNN::CV::imread(file);  // 读取图像文件
    auto st    = std::chrono::system_clock::now();  // 开始计时
    VARP image_embedding;

    // 检查是否为Qwen2-VL模型（基于输入名称判断）
    if (mul_module_->getInfo()->inputNames[0] == "patches") {
        // === Qwen2-VL 处理流程 ===
        // 调整图像尺寸为28的倍数（适配patch分割）
        image_height_ = round(image_height_ / 28.0) * 28;
        image_width_ = round(image_width_ / 28.0) * 28;
        
        // 图像预处理：缩放、颜色空间转换、归一化
        image = MNN::CV::resize(image, {image_height_, image_width_}, 0, 0,
                                     MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                     image_mean_, image_norm_);
        image = MNN::Express::_Unsqueeze(image, {0});  // 添加批次维度
        image = MNN::Express::_Convert(image, NCHW);   // 转换为NCHW格式
        
        // 构造时序patches（将单张图像复制为时序序列）
        auto patches = MNN::Express::_Concat({image, image}, 0);
        auto patches_dim = patches->getInfo()->dim;
        int temporal = patches_dim[0];  // 时序维度
        int channel  = patches_dim[1];  // 通道维度
        int height   = patches_dim[2];  // 高度维度
        int width    = patches_dim[3];  // 宽度维度
        
        // 定义patch参数
        constexpr int temporal_patch_size = 2;  // 时序patch大小
        constexpr int patch_size = 14;          // 空间patch大小
        constexpr int merge_size = 2;           // 合并大小
        
        // 计算网格维度
        int grid_t = temporal / temporal_patch_size;  // 时序网格数
        int grid_h = height / patch_size;             // 高度网格数
        int grid_w = width / patch_size;              // 宽度网格数
        
        // 构造patches：将图像分割为patch序列
        patches = MNN::Express::_Reshape(patches, {
            grid_t, temporal_patch_size,
            channel,
            grid_h / merge_size, merge_size, patch_size,
            grid_w / merge_size, merge_size, patch_size,
        });
        // 重新排列patch维度
        patches = MNN::Express::_Permute(patches, {0, 3, 6, 4, 7, 2, 1, 5, 8});
        // 展平为序列格式
        patches = MNN::Express::_Reshape(patches, {
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size
        });
        
        const int seq_len = grid_t * grid_h * grid_w;  // 序列长度
        
        // 构造位置编码（2D位置信息）
        const int wblock_size = merge_size * merge_size;
        const int hblock_size = wblock_size * grid_w / merge_size;
        VARP position_ids = MNN::Express::_Input({2, seq_len}, NCHW, halide_type_of<int>());
        auto hpos_ptr = position_ids->writeMap<int>();  // 高度位置指针
        auto wpos_ptr = hpos_ptr + seq_len;             // 宽度位置指针
        
        // 填充位置编码
        for (int i = 0; i < grid_h; i++) {
            int h_idx = i / merge_size, h_off = i % merge_size;
            for (int j = 0; j < grid_w; j++) {
                int w_idx = j / merge_size, w_off = j % merge_size;
                int index = h_idx * hblock_size + w_idx * wblock_size + h_off * 2 + w_off;
                hpos_ptr[index] = i;  // 设置高度位置
                wpos_ptr[index] = j;  // 设置宽度位置
            }
        }
        
        // 构造注意力掩码（全零，表示所有位置都可见）
        VARP attention_mask = MNN::Express::_Input({1, seq_len, seq_len}, NCHW);
        ::memset(attention_mask->writeMap<float>(), 0, seq_len * seq_len * sizeof(float));
        
        // 通过视觉编码器生成图像嵌入
        image_embedding = mul_module_->onForward({patches, position_ids, attention_mask})[0];
    } else {
        // === 标准视觉模型处理流程 ===
        // 图像预处理：缩放、颜色转换、归一化
        image = MNN::CV::resize(image, {image_height_, image_width_}, 0, 0,
                                      MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                      image_mean_, image_norm_);
        image = MNN::Express::_Unsqueeze(image, {0});  // 添加批次维度
        image = MNN::Express::_Convert(image, NC4HW4);  // 转换为NC4HW4格式（MNN优化格式）
        // 通过视觉编码器生成图像嵌入
        image_embedding = mul_module_->forward(image);
    }
    
    auto et = std::chrono::system_clock::now();  // 结束计时
    // 记录视觉处理耗时
    mState.vision_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    
    // 缓存图像嵌入供后续使用
    mul_embeddings_.push_back(image_embedding);
    
    // 构造对应的token序列
    int visual_len = image_embedding->getInfo()->dim[0];  // 视觉序列长度
    std::vector<int> img_ids(visual_len, img_pad_);       // 用填充token填充序列
    img_ids.insert(img_ids.begin(), vision_start_);       // 添加视觉开始token
    img_ids.push_back(vision_end_);                       // 添加视觉结束token
    
    return img_ids;  // 返回视觉token序列
#else
    return std::vector<int>(0);  // 如果未启用视觉支持，返回空序列
#endif
}

// 辅助模板函数：创建常量张量
template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

// 音频处理方法：将音频文件转换为token序列
// 使用Whisper的filter bank特征提取和音频编码器
std::vector<int> Mllm::audio_process(const std::string& file) {
#ifdef LLM_SUPPORT_AUDIO
    constexpr int sample_rate = 16000;  // 固定采样率16kHz，Whisper标准
    
    // 加载音频文件并重采样到指定采样率
    auto load_res = MNN::AUDIO::load(file, sample_rate);
    VARP waveform = load_res.first;   // 获取波形数据
    // int sample_rate = load_res.second;  // 实际采样率（未使用）
    
    int wav_len = waveform->getInfo()->dim[0];  // 音频长度（采样点数）
    int hop_length = 160;  // 跳跃长度（未使用，预留）
    
    auto st = std::chrono::system_clock::now();  // 开始计时
    
    // 提取Whisper风格的filter bank特征
    auto input_features = MNN::AUDIO::whisper_fbank(waveform);
    
    // 通过音频编码器生成音频嵌入
    auto audio_embedding = mul_module_->forward(input_features);
    
    // 调整嵌入张量维度：从[batch, seq_len, dim]转为[seq_len, batch, dim]
    audio_embedding = _Permute(audio_embedding, {1, 0, 2});
    
    auto et = std::chrono::system_clock::now();  // 结束计时
    // 记录音频处理耗时
    mState.audio_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    
    // 缓存音频嵌入供后续使用
    mul_embeddings_.push_back(audio_embedding);
    
    // 构造对应的token序列
    int embed_len = audio_embedding->getInfo()->dim[0];  // 音频序列长度
    std::vector<int> audio_ids(embed_len, audio_pad_);   // 用音频填充token填充序列
    
    return audio_ids;  // 返回音频token序列
#else
    return std::vector<int>(0);  // 如果未启用音频支持，返回空序列
#endif
}

// 多模态处理方法：统一处理图像和音频输入
// 支持本地文件和URL下载，以及图像尺寸参数解析
std::vector<int> Mllm::multimode_process(const std::string& mode, std::string info) {
    auto file_info = info;  // 文件信息，可能包含路径或URL
    
    // 处理图像模式下的特殊格式
    if (mode == "img") {
        // 解析图像尺寸标签 <hw>height,width</hw>
        std::regex hw_regex(R"(<hw>(.*?)</hw>)");  // 匹配 <hw>内容</hw> 格式
        std::sregex_iterator iter(info.begin(), info.end(), hw_regex);
        std::sregex_iterator end;
        file_info = "";  // 重置文件信息

        size_t currentPosition = 0;
        if (iter != end) {
            std::smatch match = *iter;
            size_t matchPosition = match.position();
            
            // 保留标签前的内容
            if (matchPosition > currentPosition) {
                file_info.append(info.substr(currentPosition, matchPosition - currentPosition));
            }

            // 解析尺寸参数：格式为 "height,width"
            std::stringstream hw_ss(match.str(1));
            char comma;
            hw_ss >> image_height_ >> comma >> image_width_;  // 读取高度、逗号、宽度
            currentPosition = matchPosition + match.length();
        }
        
        // 保留标签后的内容（通常是文件路径）
        if (currentPosition < info.length()) {
            file_info.append(info.substr(currentPosition));
        }
        
        // 调试输出（已注释）
        // std::cout << "hw: " << image_height_ << ", " << image_width_ << std::endl;
        // std::cout << "file: " << file_info << std::endl;
    }
    
    // 处理HTTP/HTTPS URL：下载远程文件
    if (file_info.substr(0, 4) == "http") {
        std::regex url_regex(R"(^https?://([^/]+)(/.*))");  // 解析URL：协议://主机/路径
        std::smatch url_match_result;
        std::string host, path;
        
        // 提取主机名和路径
        if (std::regex_search(file_info, url_match_result, url_regex) && url_match_result.size() == 3) {
            host = url_match_result[1].str();  // 主机名
            path = url_match_result[2].str();  // 路径
        }
        
        // 调试输出（已注释）
        // std::cout << host << "#" << path << std::endl;
        
        // 使用HTTP客户端下载文件
        httplib::Client cli(host);
        auto res = cli.Get(path);
        file_info = "downloaded_file";  // 设置本地文件名
        
        if (res && res->status == 200) {
            // 下载成功，保存到本地文件
            std::ofstream file(file_info, std::ios::binary);
            if (file.is_open()) {
                file.write(res->body.c_str(), res->body.size());
                std::cout << "File has been downloaded successfully." << std::endl;
                file.close();
            } else {
                std::cerr << "Unable to open file to write." << std::endl;
            }
        } else {
            // 下载失败
            std::cerr << "Failed to download file. Status code: " << (res ? res->status : 0) << std::endl;
        }
    }
    
    // 根据模式调用相应的处理方法
    if (mode == "img" && config_->is_visual()) {
        return vision_process(file_info);  // 处理图像
    }
    if (mode == "audio" && config_->is_audio()) {
        return audio_process(file_info);   // 处理音频
    }
    
    return std::vector<int>(0);  // 不支持的模式或配置，返回空序列
}

/**
 * @brief 支持多模态内容的token编码
 * 
 * 该方法重写了基类的token编码方法，支持处理包含图像或音频标记的文本，
 * 能够识别并处理多模态标记，生成对应的token序列。
 * 
 * @param query 输入的查询字符串，可能包含多模态标记
 * @param use_template 是否使用提示词模板，默认为true
 * @return 编码后的token ID序列
 */
std::vector<int> Mllm::tokenizer_encode(const std::string& query, bool use_template) {
    auto prompt = apply_prompt_template(query);
    // split query
    std::regex multimode_regex("<(img|audio)>(.*?)</\\1>");
    std::string::const_iterator searchStart(prompt.cbegin());
    std::smatch match;
    std::vector<std::string> img_infos;
    std::vector<int> ids{};

    while (std::regex_search(searchStart, prompt.cend(), match, multimode_regex)) {
        // std::cout << "img match: " << match[1].str() << std::endl;
        auto txt_ids = tokenizer_->encode(match.prefix().str());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
        auto mul_ids = multimode_process(match[1].str(), match[2].str());
        ids.insert(ids.end(), mul_ids.begin(), mul_ids.end());
        searchStart = match.suffix().first;
    }
    if (searchStart != prompt.cend()) {
        auto txt_ids = tokenizer_->encode(std::string(searchStart, prompt.cend()));
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
    }
    // printf("ids = ["); for (auto id : ids) printf("%d, ", id); printf("]\n");
    return ids;
}

/**
 * @brief 生成包含多模态内容的嵌入向量
 * 
 * 该方法重写了基类的嵌入生成方法，支持处理包含多模态内容的token序列，
 * 能够将文本token和多模态token（图像或音频）一起转换为嵌入向量。
 * 
 * @param input_ids 输入的token ID序列，可能包含多模态token
 * @return 生成的嵌入向量张量
 */
VARP Mllm::embedding(const std::vector<int>& input_ids) {
    if (input_ids.size() == 1) {
        return Llm::embedding(input_ids);
    }
    std::vector<VARP> embeddings;
    int mul_idx = 0;
    std::vector<int> cur_txt_ids;
    bool in_audio = false;
    for (int i = 0; i < input_ids.size(); i++) {
        int id = input_ids[i];
        // audio
        if (in_audio) {
            if (id == audio_pad_) {
                continue;
            } else {
                cur_txt_ids.clear();
                in_audio = false;
            }
        } else if (id == audio_pad_) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto mul_embedding = mul_embeddings_[mul_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
            in_audio = true;
        }
        // vision
        if (id == img_pad_) {
            continue;
        }
        cur_txt_ids.push_back(id);
        if (id == vision_start_) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto mul_embedding = mul_embeddings_[mul_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
        } else if (id == vision_end_) {
            cur_txt_ids.clear();
            cur_txt_ids.push_back(id);
        }
    }
    mul_embeddings_.clear();
    if (!cur_txt_ids.empty()) {
        auto txt_embedding = Llm::embedding(cur_txt_ids);
        embeddings.push_back(txt_embedding);
    }
    auto embedding = MNN::Express::_Concat(embeddings, 0);
    return embedding;
}
// Llm end

// Embedding start
/**
 * @brief 计算两个嵌入向量之间的欧几里得距离
 * 
 * 该方法用于计算两个嵌入向量之间的欧几里得距离，常用于相似度计算。
 * 
 * @param var0 第一个嵌入向量
 * @param var1 第二个嵌入向量
 * @return 两个向量之间的欧几里得距离
 */
float Embedding::dist(VARP var0, VARP var1) {
    auto distVar = _Sqrt(_ReduceSum(_Square(var0 - var1)));
    auto dist    = distVar->readMap<float>()[0];
    return dist;
}

/**
 * @brief 创建嵌入模型实例
 * 
 * 该方法用于创建嵌入模型实例，根据配置文件路径初始化模型。
 * 
 * @param config_path 配置文件路径
 * @param load 是否加载模型权重，默认为true
 * @return 创建的嵌入模型实例指针
 */
Embedding* Embedding::createEmbedding(const std::string& config_path, bool load) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Embedding* embedding = new Embedding(config);
    if (load) {
        embedding->load();
    }
    return embedding;
}

/**
 * @brief Embedding类构造函数
 * 
 * 该构造函数用于初始化嵌入模型对象。
 * 
 * @param config 模型配置指针
 */
Embedding::Embedding(std::shared_ptr<LlmConfig> config) : Llm(config) {
}

/**
 * @brief 获取嵌入维度
 * 
 * 该方法用于获取嵌入向量的维度大小。
 * 
 * @return 嵌入向量的维度
 */
int Embedding::dim() const {
    return config_->hidden_size();
}

/**
 * @brief 加载嵌入模型
 * 
 * 该方法用于加载嵌入模型的所有必要组件，包括运行时环境、分词器、磁盘嵌入和模型权重。
 */
void Embedding::load() {
    init_runtime();
    printf("load tokenizer\n");
    std::cout << config_->tokenizer_file() << std::endl;
    // 1. load vocab
    tokenizer_.reset(Tokenizer::createTokenizer(config_->tokenizer_file()));
    printf("load tokenizer Done\n");
    disk_embedding_.reset(new DiskEmbedding(config_));
    // 2. load model
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange    = true;
    auto model_path            = config_->llm_model();
    MNN_PRINT("load %s ... ", model_path.c_str());
    modules_.resize(1);
    modules_[0].reset(Module::load({"input_ids", "attention_mask", "position_ids"}, {"sentence_embeddings"},
                                   model_path.c_str(), runtime_manager_, &module_config));
    MNN_PRINT("Done!\n");
}

/**
 * @brief 根据token ID序列计算嵌入向量
 * 
 * 该方法用于根据输入的token ID序列计算对应的嵌入向量，
 * 通过模型前向推理获得句子级别的嵌入表示。
 * 
 * @param ids 输入的token ID序列
 * @return 计算得到的嵌入向量张量
 */
VARP Embedding::ids_embedding(const std::vector<int>& ids) {
    int prompt_len           = ids.size();
    auto inputs_ids          = embedding(ids);
    auto attention_mask      = gen_attention_mask(prompt_len);
    auto position_ids        = gen_position_ids(prompt_len);
    auto outputs             = modules_[0]->onForward({inputs_ids, attention_mask, position_ids});
    auto sentence_embeddings = outputs[0];
    return sentence_embeddings;
}

/**
 * @brief 根据文本计算嵌入向量
 * 
 * 该方法用于根据输入的文本计算对应的嵌入向量，
 * 先将文本编码为token ID序列，再计算嵌入向量。
 * 
 * @param txt 输入的文本
 * @return 计算得到的嵌入向量张量
 */
VARP Embedding::txt_embedding(const std::string& txt) {
    return ids_embedding(tokenizer_encode(txt));
}

/**
 * @brief 生成嵌入计算用的注意力掩码
 * 
 * 该方法用于生成嵌入计算所需的注意力掩码张量，
 * 与LLM类中的同名方法不同，这里生成的是全1掩码。
 * 
 * @param seq_len 序列长度
 * @return 生成的注意力掩码张量
 */
VARP Embedding::gen_attention_mask(int seq_len) {
    auto attention_mask = _Input({1, 1, 1, seq_len}, NCHW, halide_type_of<int>());
    auto ptr            = attention_mask->writeMap<int>();
    for (int i = 0; i < seq_len; i++) {
        ptr[i] = 1;
    }
    return attention_mask;
}

/**
 * @brief 生成嵌入计算用的位置ID
 * 
 * 该方法用于生成嵌入计算所需的位置ID张量，
 * 与LLM类中的同名方法不同，这里生成的是简单的递增序列。
 * 
 * @param seq_len 序列长度
 * @return 生成的位置ID张量
 */
VARP Embedding::gen_position_ids(int seq_len) {
    auto position_ids = _Input({1, seq_len}, NCHW, halide_type_of<int>());
    auto ptr          = position_ids->writeMap<int>();
    for (int i = 0; i < seq_len; i++) {
        ptr[i] = i;
    }
    return position_ids;
}
// Embedding end
} // namespace Transformer
} // namespace MNN
