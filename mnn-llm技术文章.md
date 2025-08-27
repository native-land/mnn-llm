# 使用MNN部署语言模型

## 1. 引言

[英伟达的论文](https://arxiv.org/html/2506.02153v1)指出，在"代理型 AI"（Agentic AI）场景中，
Small Language Models（SLMs） 足够强大、运算合适且更加经济，因此比大型语言模型（LLMs）更适合作为未来趋势；
当需要通用对话能力时，推荐 异构系统（结合 SLM 与 LLM 的模块化系统），小型语言模型在实际应用中的巨大潜力。

越来越多的实际应用场景需要在终端设备上部署语言模型：

**终端AI助手**
- 低延迟响应：本地推理避免网络延迟
- 离线运行：无需网络连接即可工作
- 隐私保护：敏感数据不离开本地设备

**边缘计算**
- IoT设备部署：在资源受限的嵌入式设备上运行
- 实时推理：工业控制、自动驾驶等需要实时响应的场景
- 资源受限环境：电力、带宽、计算资源有限的环境

**移动应用**
- Android/iOS原生应用：集成到移动App中
- 本地知识问答：无需联网的智能问答系统
- 实时对话系统：语音助手、客服机器人等

面对这些应用需求，阿里开源的MNN推理框架为我们提供了一个出色的解决方案。

本文将基于 [mnn-llm](https://github.com/wangzhaode/mnn-llm) 这个使用MNN框架部署大语言模型的实际案例，分析如何使用MNN框架实现大语言模型的终端部署。

## 2. 核心业务流程梳理

使用MNN部署大语言模型涉及三个主要流程：**模型加载流程**、**推理生成流程**和**CLI应用流程**。

### 2.1 模型加载流程

模型加载是整个系统的初始化过程：

```cpp
// 核心流程：配置解析 → MNN模型加载 → 运行时初始化
void Llm::load() {
    // 1. 获取配置文件中的模型路径
    auto model_path = config_->llm_model();
    
    // 2. 使用MNN加载模型文件
    modules_.emplace_back(Module::load({}, {}, model_path.c_str()));
    
    // 3. 初始化推理运行时环境
    init_runtime();
}
```

**关键步骤**：
- 从JSON配置文件读取模型文件路径
- 调用MNN的`Module::load()`加载.mnn模型文件
- 配置推理后端（CPU/GPU/NPU）和线程数

### 2.2 推理生成流程

这是核心的文本生成流程，从用户输入到模型输出：

```cpp
// 核心流程：文本预处理 → MNN推理 → 结果后处理
void response(const std::string& user_content) {
    // 1. 应用提示词模板
    auto prompt = apply_prompt_template(user_content);
    
    // 2. 文本分词转换为token序列
    auto input_ids = tokenizer_encode(prompt);
    
    // 3. 自回归生成token序列
    auto output_ids = generate(input_ids);
    
    // 4. 将token解码为文本输出
    for(int id : output_ids) {
        std::cout << tokenizer_decode(id);
    }
}
```

**关键步骤**：
- 输入文本按提示词模板格式化
- Tokenizer将文本转换为模型可理解的数字序列
- 模型自回归生成新的token
- 将生成的token解码回文本

### 2.3 CLI应用流程

CLI Demo展示了完整的应用流程：

```cpp
// 核心流程：参数解析 → 模型初始化 → 业务执行
int main(int argc, const char* argv[]) {
    // 1. 解析命令行参数
    std::string model_dir = argv[1];
    
    // 2. 初始化LLM实例
    std::unique_ptr<Llm> llm(Llm::createLLM(model_dir));
    llm->load();
    
    // 3. 选择执行模式
    if (argc < 3) {
        llm->chat();  // 交互式对话
    } else {
        benchmark(llm.get(), argv[2]);  // 批量测试
    }
}
```

**关键步骤**：
- 解析模型目录路径参数
- 创建并加载LLM实例
- 根据参数选择交互模式或测试模式

## 3. 核心技术实现细节

本章深入分析MNN-LLM项目中的关键技术实现，重点关注文本处理、MNN接口使用和数据转换等核心环节。

### 3.1 文本分词技术实现

#### Tokenizer的工作机制

Tokenizer负责文本和token序列之间的双向转换：

```cpp
// include/tokenizer.hpp
class Tokenizer {
public:
    // 文本编码：文本 → token序列
    std::vector<int> encode(const std::string& str) {
        // 1. 文本预处理（标准化、清洗）
        auto processed_text = preprocess(str);
        
        // 2. 应用分词算法（BPE/SentencePiece/WordPiece）
        auto tokens = tokenize(processed_text);
        
        // 3. 转换为数字ID
        std::vector<int> token_ids;
        for (const auto& token : tokens) {
            token_ids.push_back(vocab_[token]);
        }
        return token_ids;
    }
    
    // token解码：数字ID → 文本
    virtual std::string decode(int id) = 0;
    
    // 特殊token判断
    bool is_stop(int token) { return stop_words_.count(token) > 0; }
    bool is_special(int token) { return special_tokens_.count(token) > 0; }
};
```

#### 多种Tokenizer实现

项目支持主流的分词器类型：

```cpp
// SentencePiece分词器 (src/tokenizer.cpp:245-280)
class SentencePieceTokenizer : public Tokenizer {
    sentencepiece::SentencePieceProcessor sp_;
    
    std::vector<int> encode(const std::string& str) override {
        std::vector<int> ids;
        sp_.Encode(str, &ids);  // 使用SentencePiece库编码
        return ids;
    }
    
    std::string decode(int id) override {
        return sp_.IdToPiece(id);  // ID转换为piece
    }
};

// TikToken分词器 (用于GPT系列模型)
class TikTokenizer : public Tokenizer {
    tiktoken::Encoding enc_;
    
    std::vector<int> encode(const std::string& str) override {
        return enc_.encode(str);  // 使用TikToken编码
    }
    
    std::string decode(int id) override {
        return enc_.decode_single_token(id);
    }
};
```

### 3.2 词嵌入技术实现

#### 嵌入向量的生成过程

```cpp
// src/llm.cpp中的embedding实现
MNN::Express::VARP embedding(const std::vector<int>& input_ids) {
    // 1. 从token ID查找对应的嵌入向量
    if (disk_embedding_) {
        // 磁盘嵌入：按需从磁盘加载
        auto embedding_ptr = std::make_shared<float>(input_ids.size() * hidden_size_);
        disk_embedding_->disk_embedding_lookup(input_ids, embedding_ptr.get());
        
        // 2. 创建MNN Tensor
        auto embedding_tensor = MNN::Express::_Const(
            embedding_ptr.get(), 
            {(int)input_ids.size(), hidden_size_}, 
            MNN::Express::NHWC
        );
        return embedding_tensor;
    } else {
        // 内存嵌入：直接从权重矩阵查找
        return embedding_weight_[input_ids];  // 简化表示
    }
}
```

#### 磁盘嵌入优化实现

为了节省内存，大型嵌入矩阵可以存储在磁盘上：

```cpp
// src/llm.cpp:99-228 DiskEmbedding类
class DiskEmbedding {
private:
    std::unique_ptr<uint8_t[]> weight_;      // 权重数据缓冲区
    std::unique_ptr<uint8_t[]> alpha_;       // 量化参数缓冲区
    int hidden_size_;                        // 嵌入维度
    int quant_bit_;                          // 量化位数（4位或8位）
    
public:
    void disk_embedding_lookup(const std::vector<int>& input_ids, float* dst) {
        for (size_t i = 0; i < input_ids.size(); i++) {
            int token = input_ids[i];
            
            if (quant_bit_ > 0) {
                // 量化模式：从磁盘读取量化数据并反量化
                seek_read(weight_.get(), weight_token_size_, 
                         w_offset_ + token * weight_token_size_);
                
                // 按块反量化
                auto alpha_ptr = reinterpret_cast<float*>(alpha_.get()) 
                               + token * block_num_ * 2;
                for (int n = 0; n < block_num_; n++) {
                    float scale = alpha_ptr[n * 2 + 1];  // 缩放因子
                    float zero = alpha_ptr[n * 2];       // 零点
                    uint8_t* src = weight_.get() + n * (quant_block_ * quant_bit_ / 8);
                    float* dst_ptr = dst + i * hidden_size_ + n * quant_block_;
                    
                    // 4位或8位反量化
                    dequant_(src, dst_ptr, scale, zero, quant_block_);
                }
            } else {
                // bf16模式：直接读取bf16数据
                seek_read(weight_.get(), weight_token_size_, token * weight_token_size_);
                bf16_to_fp32(weight_.get(), dst + i * hidden_size_, hidden_size_);
            }
        }
    }
};
```

### 3.3 MNN推理接口使用

#### MNN模型加载

```cpp
// src/llm.cpp中的模型加载过程
void Llm::load() {
    // 1. 配置MNN运行时参数
    MNN::ScheduleConfig config;
    config.type = backend_type_convert(config_->backend_type());  // CPU/GPU/NPU
    config.numThread = config_->thread_num();                     // 线程数
    
    // 2. 加载MNN模型文件
    auto model_path = config_->llm_model();
    auto runtime_manager = MNN::Express::ExecutorScope::Current()->getRuntime();
    
    // 3. 创建Module实例
    modules_.emplace_back(MNN::Module::load(
        {"input_ids", "attention_mask", "position_ids"},  // 输入名称
        {"output"},                                        // 输出名称
        model_path.c_str(),                               // 模型路径
        runtime_manager,                                   // 运行时
        &config                                           // 配置
    ));
}
```

#### MNN推理执行

```cpp
// MNN前向推理的实现
MNN::Express::VARP forward(const std::vector<int>& input_ids) {
    // 1. 准备输入数据
    auto embeddings = embedding(input_ids);              // 词嵌入
    auto attention_mask = gen_attention_mask(input_ids); // 注意力掩码
    auto position_ids = gen_position_ids(input_ids);     // 位置编码
    
    // 2. 调用MNN推理
    auto outputs = modules_[0]->onForward({
        {"input_embeddings", embeddings},
        {"attention_mask", attention_mask},
        {"position_ids", position_ids}
    });
    
    // 3. 获取输出logits
    return outputs[0];  // shape: [batch_size, seq_len, vocab_size]
}
```

### 3.4 MNN输出处理和转换

#### Logits到Token的转换

```cpp
// 从模型输出logits中采样下一个token
int sample_token(MNN::Express::VARP logits, float temperature = 1.0) {
    // 1. 获取最后一个位置的logits
    auto last_logits = logits[logits->getInfo()->dim[1] - 1];  // [vocab_size]
    
    // 2. 应用温度参数
    if (temperature != 1.0) {
        last_logits = last_logits / temperature;
    }
    
    // 3. 计算softmax概率分布
    auto probs = MNN::Express::_Softmax(last_logits, 0);
    
    // 4. 采样策略选择
    if (do_sample_) {
        // 随机采样
        return multinomial_sample(probs);
    } else {
        // 贪婪搜索：选择概率最大的token
        auto max_indices = MNN::Express::_ArgMax(probs, 0);
        return max_indices->readMap<int>()[0];
    }
}
```

#### 自回归生成循环

```cpp
// 文本生成的完整流程
std::vector<int> generate(const std::vector<int>& input_ids) {
    std::vector<int> generated_ids = input_ids;
    
    for (int step = 0; step < max_new_tokens_; step++) {
        // 1. MNN模型推理
        auto logits = forward(generated_ids);
        
        // 2. 采样下一个token
        int next_token = sample_token(logits, temperature_);
        
        // 3. 检查停止条件
        if (tokenizer_->is_stop(next_token)) {
            break;
        }
        
        // 4. 添加到序列中
        generated_ids.push_back(next_token);
        
        // 5. 实时输出（流式生成）
        std::cout << tokenizer_->decode(next_token) << std::flush;
    }
    
    return generated_ids;
}
```

### 3.5 性能监控实现

```cpp
// 生成状态统计
struct GenerateState {
    int prompt_len_ = 0;          // 提示词长度
    int gen_seq_len_ = 0;         // 生成序列长度  
    int64_t prefill_us_ = 0;      // 预填充耗时(微秒)
    int64_t decode_us_ = 0;       // 解码耗时(微秒)
    
    // 计算性能指标
    float prefill_speed() const {
        return prompt_len_ / (prefill_us_ / 1e6f);  // tokens/秒
    }
    
    float decode_speed() const {
        return gen_seq_len_ / (decode_us_ / 1e6f);  // tokens/秒
    }
};

// 性能统计实现
void benchmark_performance(const GenerateState& state) {
    printf("=== 性能报告 ===\n");
    printf("提示词长度: %d tokens\n", state.prompt_len_);
    printf("生成长度: %d tokens\n", state.gen_seq_len_);
    printf("预填充速度: %.2f tok/s\n", state.prefill_speed());
    printf("生成速度: %.2f tok/s\n", state.decode_speed());
    printf("总耗时: %.2f ms\n", (state.prefill_us_ + state.decode_us_) / 1000.0);
}
```

## 4. 使用MNN的核心价值

通过业务流程分析，MNN框架为LLM部署提供了：

**🔧 MNN框架提供**：
- 高效的神经网络推理引擎
- 跨平台的硬件适配（CPU/GPU/NPU）
- 内存和计算资源优化
- 模型加载和执行管理

**💼 开发者专注**：
- 业务逻辑设计（对话管理、用户交互）
- 数据预处理（分词、格式转换）
- 应用层优化（缓存策略、性能监控）
- 用户体验（命令行界面、流式输出）

这种分工让开发者可以**专注业务创新**，无需关心底层推理引擎的复杂实现。

## 5. 总结

通过对MNN-LLM项目的深入分析，我们可以看到使用MNN框架部署大语言模型的完整技术路径：

### 5.1 核心业务流程清晰简洁

项目展示了三个核心流程的简洁实现：
- **模型加载流程**：从配置文件到MNN运行时的标准化初始化过程
- **推理生成流程**：文本预处理 → MNN推理 → 结果后处理的完整链路
- **CLI应用流程**：从命令行参数到业务执行的用户友好界面

### 5.2 技术实现务实高效

在技术实现层面，项目采用了多项实用技术：
- **多样化分词支持**：SentencePiece、TikToken等主流分词器，适配不同模型需求
- **灵活的嵌入处理**：支持内存和磁盘两种嵌入模式，在性能和内存之间找到平衡
- **标准的MNN接口**：直接使用MNN::Module的标准API，降低学习和维护成本
- **高效的输出转换**：从logits到文本的完整转换链路，支持多种采样策略

### 5.3 MNN框架的实用价值

通过这个Demo项目，我们看到MNN框架的核心价值：
- **开发效率高**：开发者只需关注业务逻辑，无需处理底层推理优化
- **部署门槛低**：统一的接口设计，简化了模型加载和推理过程
- **性能表现好**：内置的量化、缓存等优化技术，适合资源受限环境
- **平台兼容强**：支持多种硬件后端，一套代码多平台部署

### 5.4 实际应用启示

作为一个实用的Demo项目，MNN-LLM为开发者提供了宝贵的实践参考：

1. **技术选型**：展示了如何选择合适的分词器、嵌入方案和推理策略
2. **性能优化**：通过磁盘嵌入、量化等技术实现内存和性能的平衡
3. **工程实践**：从配置管理到性能监控的完整工程化实现
4. **用户体验**：交互式对话和批量测试两种模式，满足不同使用场景

随着边缘AI和终端部署需求的增长，MNN-LLM这样的轻量级解决方案将发挥越来越重要的作用。对于希望在资源受限环境中部署大语言模型的开发者来说，这个项目提供了一个优秀的技术实现参考。

---
*本文基于 [mnn-llm](https://github.com/wangzhaode/mnn-llm) 开源项目源码分析，该项目是使用MNN框架部署大语言模型的优秀实践案例*