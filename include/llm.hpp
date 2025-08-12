//
//  llm.hpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//

#ifndef LLM_hpp
#define LLM_hpp

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <streambuf>
#include <functional>
#include <unordered_map>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

namespace MNN {
namespace Transformer {
class Tokenizer;
class Pipeline;
class LlmConfig;
class DiskEmbedding;

/**
 * @brief 微调类型枚举
 */
enum TuneType {
    // op encoder number for commit
    OP_ENCODER_NUMBER = 0,
};
struct KVMeta;
/**
 * @brief 大语言模型基类，封装了模型加载、推理、聊天等核心功能
 */
class MNN_PUBLIC Llm {
    using PromptItem = std::pair<std::string, std::string>; // <role, content>
public:
    /**
     * @brief 模型推理阶段枚举
     */
    enum Stage {
        Prefill, ///< 预填充阶段，处理用户输入并生成第一个输出token
        Decode   ///< 解码阶段，基于历史输出逐个生成后续token
    };
    /**
     * @brief 生成状态结构体，记录生成过程中的各种信息
     */
    struct GenerateState {
        // forward info
        int prompt_len_ = 0;          ///< 提示词长度
        int gen_seq_len_ = 0;         ///< 生成序列长度
        int all_seq_len_ = 0;         ///< 总序列长度 (prompt + generated)
        std::vector<int> history_ids_;///< 历史token ID序列
        // time
        int64_t vision_us_ = 0;       ///< 视觉处理耗时 (微秒)
        int64_t audio_us_ = 0;        ///< 音频处理耗时 (微秒)
        int64_t prefill_us_ = 0;      ///< Prefill阶段耗时 (微秒)
        int64_t decode_us_ = 0;       ///< Decode阶段耗时 (微秒)
        int current_token_ = 0;       ///< 当前生成的token ID
        std::vector<int> output_ids_; ///< 输出token ID序列
        std::ostream* os_ = nullptr;  ///< 输出流指针
        std::string end_with_;        ///< 结束标记
    };
    /**
     * @brief 构造函数
     * @param config 模型配置指针
     */
    Llm(std::shared_ptr<LlmConfig> config);
    /**
     * @brief 虚析构函数
     */
    virtual ~Llm();
    /**
     * @brief 根据配置文件路径创建LLM实例
     * @param config_path 配置文件路径
     * @return LLM实例指针
     */
    static Llm* createLLM(const std::string& config_path);
    /**
     * @brief 启动聊天交互
     */
    void chat();
    /**
     * @brief 重置模型状态
     */
    void reset();
    /**
     * @brief 开启或关闭性能追踪
     * @param start true为开启，false为关闭
     */
    void trace(bool start);
    /**
     * @brief 微调模型参数
     * @param type 微调类型
     * @param candidates 候选参数列表
     */
    void tuning(TuneType type, std::vector<int> candidates);
    /**
     * @brief 加载模型权重
     */
    virtual void load();
    /**
     * @brief 切换推理阶段
     * @param stage 目标阶段 (Prefill/Decode)
     */
    void switchMode(Stage stage);
    /**
     * @brief 原始前向推理接口
     * @param hiddenState 隐藏状态输入
     * @param mask 注意力掩码
     * @param inputPos 输入位置信息
     * @return 推理结果
     */
    MNN::Express::VARP forwardRaw(MNN::Express::VARP hiddenState, MNN::Express::VARP mask, MNN::Express::VARP inputPos);
    /**
     * @brief 生成注意力掩码
     * @param seq_len 序列长度
     * @return 注意力掩码变量
     */
    virtual MNN::Express::VARP gen_attention_mask(int seq_len);
    /**
     * @brief 根据输入ID生成嵌入向量
     * @param input_ids 输入token ID序列
     * @return 嵌入向量
     */
    virtual MNN::Express::VARP embedding(const std::vector<int>& input_ids);

    /**
     * @brief 前向推理接口
     * @param input_ids 输入token ID序列
     * @return 推理结果
     */
    MNN::Express::VARP forward(const std::vector<int>& input_ids);
    /**
     * @brief 从logits中采样下一个token
     * @param logits 模型输出的logits
     * @param pre_ids 之前的token ID序列
     * @param offset 采样起始偏移
     * @param size 采样范围大小
     * @return 采样得到的token ID
     */
    int sample(MNN::Express::VARP logits, const std::vector<int>& pre_ids, int offset = 0, int size = 0);
    /**
     * @brief 应用提示词模板
     * @param user_content 用户输入内容
     * @return 应用模板后的字符串
     */
    std::string apply_prompt_template(const std::string& user_content) const;
    /**
     * @brief 应用聊天模板
     * @param chat_prompts 聊天历史记录
     * @return 应用模板后的字符串
     */
    std::string apply_chat_template(const std::vector<PromptItem>& chat_prompts) const;
    /**
     * @brief 获取当前历史记录长度
     * @return 历史记录长度
     */
    size_t getCurrentHistory() const;
    /**
     * @brief 删除指定范围的历史记录
     * @param begin 删除起始位置
     * @param end 删除结束位置
     */
    void eraseHistory(size_t begin, size_t end);
    /**
     * @brief 对用户输入进行响应
     * @param user_content 用户输入内容
     * @param os 输出流指针，默认为标准输出
     * @param end_with 结束标记，默认为nullptr
     * @param max_new_tokens 最大生成token数，默认为-1 (无限制)
     */
    void response(const std::string& user_content, std::ostream* os = &std::cout, const char* end_with = nullptr, int max_new_tokens = -1);
    /**
     * @brief 对聊天历史进行响应
     * @param chat_prompts 聊天历史记录
     * @param os 输出流指针，默认为标准输出
     * @param end_with 结束标记，默认为nullptr
     * @param max_new_tokens 最大生成token数，默认为-1 (无限制)
     */
    void response(const std::vector<PromptItem>& chat_prompts, std::ostream* os = &std::cout, const char* end_with = nullptr, int max_new_tokens = -1);
    /**
     * @brief 初始化生成过程
     * @param os 输出流指针，默认为nullptr
     * @param end_with 结束标记，默认为nullptr
     */
    void generate_init(std::ostream* os = nullptr, const char* end_with = nullptr);
    /**
     * @brief 生成指定数量的token
     * @param max_token 最大生成token数
     */
    void generate(int max_token);
    /**
     * @brief 根据输入ID生成token序列
     * @param input_ids 输入token ID序列
     * @param max_new_tokens 最大生成token数，默认为-1 (无限制)
     * @return 生成的token ID序列
     */
    std::vector<int> generate(const std::vector<int>& input_ids, int max_new_tokens = -1);
    /**
     * @brief 判断是否停止生成
     * @return true表示停止，false表示继续
     */
    bool stoped();
    /**
     * @brief 打印推理速度信息
     */
    void print_speed();
    // config function
    /**
     * @brief 导出模型配置信息
     * @return 配置信息字符串
     */
    std::string dump_config();
    /**
     * @brief 设置模型配置
     * @param content 配置内容字符串
     * @return true表示设置成功，false表示失败
     */
    bool set_config(const std::string& content);
    // lora function
    /**
     * @brief 应用LoRA权重
     * @param lora_path LoRA权重路径
     * @return LoRA模块索引
     */
    size_t apply_lora(const std::string& lora_path);
    /**
     * @brief 创建LoRA模型实例
     * @param lora_path LoRA权重路径
     * @return LoRA模型实例指针
     */
    Llm* create_lora(const std::string& lora_path);
    /**
     * @brief 释放指定索引的模块
     * @param index 模块索引
     * @return true表示释放成功，false表示失败
     */
    bool release_module(size_t index);
    /**
     * @brief 选择指定索引的模块
     * @param index 模块索引
     * @return true表示选择成功，false表示失败
     */
    bool select_module(size_t index);
    // tokenier function
    /**
     * @brief 判断token ID是否为停止符
     * @param token_id token ID
     * @return true表示是停止符，false表示不是
     */
    bool is_stop(int token_id);
    /**
     * @brief 根据token ID解码为文本
     * @param id token ID
     * @return 解码后的文本
     */
    std::string tokenizer_decode(int id);
    /**
     * @brief 将文本编码为token ID序列
     * @param query 输入文本
     * @param use_template 是否使用提示词模板
     * @return token ID序列
     */
    virtual std::vector<int> tokenizer_encode(const std::string& query, bool use_template = true);
    friend class Pipeline;
    const GenerateState& getState() const {
        return mState;
    }
protected:
    std::shared_ptr<KVMeta> mMeta; ///< KV缓存元数据
    std::shared_ptr<LlmConfig> config_; ///< 模型配置
    std::shared_ptr<Tokenizer> tokenizer_; ///< 分词器
    std::shared_ptr<DiskEmbedding> disk_embedding_; ///< 磁盘嵌入
    MNN::Express::VARP inputs_embeds_, attention_mask_, position_ids_; ///< 输入嵌入、注意力掩码、位置ID
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime_manager_; ///< 运行时管理器
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> mllm_runtime_manager_; ///< MLLM运行时管理器
    std::vector<std::shared_ptr<MNN::Express::Module>> modules_; ///< 所有模块
    std::vector<std::shared_ptr<MNN::Express::Module>> prefill_modules_, decode_modules_, current_modules_; ///< Prefill模块、Decode模块、当前模块
    const MNN::Express::Module* base_module_ = nullptr; ///< 基础模块指针
    void init_runtime(); ///< 初始化运行时环境
    virtual MNN::Express::VARP gen_position_ids(int seq_len); ///< 生成位置ID
    bool mTracing = false; ///< 是否开启性能追踪
    GenerateState mState; ///< 生成状态
};

// Embedding start
/**
 * @brief 嵌入模型类，继承自Llm类，用于文本嵌入计算
 */
class MNN_PUBLIC Embedding : public Llm {
public:
    /**
     * @brief 构造函数
     * @param config 模型配置指针
     */
    Embedding(std::shared_ptr<LlmConfig> config);
    /**
     * @brief 根据配置文件路径创建嵌入模型实例
     * @param config_path 配置文件路径
     * @param load 是否加载模型权重，默认为true
     * @return 嵌入模型实例指针
     */
    static Embedding* createEmbedding(const std::string& config_path, bool load = true);
    /**
     * @brief 计算两个嵌入向量之间的距离
     * @param var0 第一个嵌入向量
     * @param var1 第二个嵌入向量
     * @return 向量间距离
     */
    static float dist(MNN::Express::VARP var0, MNN::Express::VARP var1);
    /**
     * @brief 加载模型权重
     */
    virtual void load() override;
    /**
     * @brief 根据token ID序列计算嵌入向量
     * @param ids token ID序列
     * @return 嵌入向量
     */
    MNN::Express::VARP ids_embedding(const std::vector<int>& ids);
    /**
     * @brief 根据文本计算嵌入向量
     * @param txt 输入文本
     * @return 嵌入向量
     */
    MNN::Express::VARP txt_embedding(const std::string& txt);
    /**
     * @brief 获取嵌入维度
     * @return 嵌入维度
     */
    int dim() const;
private:
    virtual MNN::Express::VARP gen_attention_mask(int seq_len) override; ///< 生成注意力掩码
    virtual MNN::Express::VARP gen_position_ids(int seq_len) override; ///< 生成位置ID
};
// Embedding end
}
}

#endif // LLM_hpp
