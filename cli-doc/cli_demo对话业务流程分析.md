# cli_demo 与 LLM 对话业务流程分析

## 主要业务流程

### 1. 程序启动和初始化
```
main() → Llm::createLLM() → llm->load()
```
- `main()` (demo/cli_demo.cpp:122): 程序入口
- 创建LLM实例并加载模型权重、分词器、配置

### 2. 对话模式选择
- **交互模式**: `llm->chat()` (src/llm.cpp:907)
- **基准测试模式**: `benchmark()` (demo/cli_demo.cpp:24)

### 3. 核心对话流程
```
response() → tokenizer_encode() → generate() → 
prefill阶段 → decode阶段 → tokenizer_decode()
```

## 直接调用的 MNN API 位置

### 核心 MNN API 调用

#### 1. 运行时管理 (src/llm.cpp:484-527)
```cpp
// 创建运行时管理器
runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));

// 设置运行时提示参数
runtime_manager_->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
runtime_manager_->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, 1);
runtime_manager_->setHint(MNN::Interpreter::QKV_QUANT_OPTIONS, config_->quant_qkv());
runtime_manager_->setHint(MNN::Interpreter::KVCACHE_SIZE_LIMIT, config_->kvcache_limit());
runtime_manager_->setHint(MNN::Interpreter::MMAP_FILE_SIZE, file_size_m(config_->llm_weight()) + 128);
runtime_manager_->setHint(MNN::Interpreter::USE_CACHED_MMAP, 1);
```

#### 2. 模块加载 (src/llm.cpp:562-564)
```cpp
modules_[0].reset(Module::load(
    {"input_ids", "attention_mask", "position_ids"}, 
    {"logits"}, model_path.c_str(), runtime_manager_, &module_config));
```

#### 3. 前向推理 (src/llm.cpp:760)
```cpp
outputs = current_modules_.back()->onForward({hiddenState, mask, inputPos});
```

#### 4. 张量操作 (src/llm.cpp:1314-1316)
```cpp
VARP res = _Input({seq_len, 1, hidden_size}, NCHW);
// 创建输入张量用于嵌入存储
```

#### 5. 注意力掩码和位置ID生成
- `gen_attention_mask()` (src/llm.cpp:1346): 生成注意力掩码张量
- `gen_position_ids()` (src/llm.cpp:1403): 生成位置ID张量

## 词嵌入实现位置

### 1. 磁盘嵌入类 (src/llm.cpp:105-228)
```cpp
class DiskEmbedding {
    void embedding(const std::vector<int>& input_ids, float* ptr);
    // 支持量化和非量化两种格式
    // 构造函数：DiskEmbedding(const std::shared_ptr<LlmConfig>& config)
    // 核心方法：embedding() - 将token ID转换为嵌入向量
}
```

### 2. 嵌入生成过程 (src/llm.cpp:1310-1318)
```cpp
VARP Llm::embedding(const std::vector<int>& input_ids) {
    int hidden_size = config_->hidden_size();
    int seq_len = static_cast<int>(input_ids.size());
    VARP res = _Input({seq_len, 1, hidden_size}, NCHW);
    // 使用磁盘嵌入将token转换为向量
    disk_embedding_->embedding(input_ids, res->writeMap<float>());
    return res;
}
```

### 3. 词嵌入特点
- **节省内存**: 使用磁盘存储而非内存加载完整词汇表嵌入
- **支持量化**: 支持4位/8位量化存储格式 (src/llm.cpp:77-96)
  - `q41_dequant_ref()`: 4位量化反量化函数
  - `q81_dequant_ref()`: 8位量化反量化函数
- **多模态支持**: 在Mllm类中扩展支持图像和音频嵌入 (src/llm.cpp:1888-1935)

### 4. 量化嵌入处理流程
```
token ID → 磁盘文件读取 → 量化数据 → 反量化 → float嵌入向量
```

## 详细对话流程步骤

### Phase 1: 初始化阶段
1. **加载运行时** - `init_runtime()` (src/llm.cpp:450)
2. **加载分词器** - `Tokenizer::createTokenizer()` (src/llm.cpp:541)
3. **初始化磁盘嵌入** - `new DiskEmbedding(config)` (src/llm.cpp:543)
4. **加载模型模块** - `Module::load()` (src/llm.cpp:562)
5. **克隆解码模块** - `Module::clone()` (src/llm.cpp:568)

### Phase 2: 对话处理阶段
1. **文本编码**: `tokenizer_encode()` → token IDs
2. **嵌入生成**: `embedding()` → 嵌入向量 (调用DiskEmbedding)
3. **预填充推理**: `forward()` → logits (处理完整输入序列)
4. **采样**: `sample()` → 下一个token ID
5. **解码循环**: 逐个生成token直到停止条件
6. **文本解码**: `tokenizer_decode()` → 输出文本

### Phase 3: KV缓存管理
- **KVMeta结构** (src/llm.cpp:42-71): 管理键值对缓存状态
- **缓存同步**: `sync()` 方法更新序列长度并重置临时变量
- **内存优化**: 支持内存映射和缓存限制

## 性能优化特性

### 1. 模块复用
- **prefill_modules_**: 预填充阶段专用模块
- **decode_modules_**: 解码阶段专用模块
- **模块切换**: `switchMode()` 在不同阶段间切换

### 2. 内存优化
- **磁盘嵌入**: 避免加载完整词汇表到内存
- **内存映射**: `USE_CACHED_MMAP` 支持文件内存映射
- **KV缓存限制**: `KVCACHE_SIZE_LIMIT` 控制缓存大小

### 3. 量化支持
- **动态量化**: `DYNAMIC_QUANT_OPTIONS` 运行时量化
- **QKV量化**: 专门的注意力权重量化
- **嵌入量化**: 4位/8位量化嵌入存储

## 多模态扩展

### Mllm类扩展 (src/llm.cpp:236-351)
- **视觉处理**: `vision_process()` - 图像转token序列
- **音频处理**: `audio_process()` - 音频转token序列  
- **多模态嵌入**: `Mllm::embedding()` - 混合文本和多模态嵌入

### 支持的输入格式
- **图像**: `<img>path</img>` 或 `<hw>height,width</hw>path`
- **音频**: `<audio>path</audio>`
- **URL下载**: 支持HTTP/HTTPS远程文件下载

## 关键流程总结

完整的对话处理流程：
```
用户输入文本
    ↓
tokenizer编码 → token IDs
    ↓
磁盘嵌入 → 嵌入向量 (调用MNN张量API)
    ↓
MNN模块前向推理 → logits
    ↓
采样算法 → 下一个token ID
    ↓
tokenizer解码 → 输出文本
    ↓
重复解码阶段直到停止条件
```

这个流程在预填充阶段处理完整输入序列，在解码阶段逐个生成token，通过KV缓存优化和模块复用实现高效推理。