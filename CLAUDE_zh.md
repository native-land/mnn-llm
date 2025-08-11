# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在处理此代码库时提供指导。

## 项目概述

此代码库包含 mnn-llm，这是一个使用 MNN 推理引擎运行大语言模型 (LLM) 的项目。该项目支持多个平台，包括 Linux、macOS、Windows、Android 和 iOS。它提供了命令行、Web、Android 和 iOS 接口的实现。

该项目已合并到 MNN 主仓库中，地址为 https://github.com/alibaba/MNN/tree/master/transformers/llm。

## 构建命令

### 本地编译
```bash
# Linux/macOS
./script/build.sh

# Windows (MSVC)
./script/build.ps1

# Python wheel
./script/py_build.sh

# Android
./script/android_build.sh

# Android APK
./script/android_app_build.sh

# iOS
./script/ios_build.sh
```

### 关键构建选项
- `BUILD_FOR_ANDROID`: 为 Android 设备编译
- `LLM_SUPPORT_VISION`: 启用视觉处理功能
- `DUMP_PROFILE_INFO`: 每次对话后转储性能数据
- 后端选项: `-DMNN_CUDA=ON`, `-DMNN_OPENCL=ON`, `-DMNN_METAL=ON`

## 执行命令

### Linux/macOS
```bash
# CLI 演示
./cli_demo ./Qwen2-1.5B-Instruct-MNN/config.json

# Web UI 演示
./web_demo ./Qwen2-1.5B-Instruct-MNN/config.json ../web
```

### Windows
```bash
# CLI 演示
.\Debug\cli_demo.exe ./Qwen2-1.5B-Instruct-MNN/config.json

# Web UI 演示
.\Debug\web_demo.exe ./Qwen2-1.5B-Instruct-MNN/config.json ../web
```

## 代码架构

### 核心组件
1. **Llm 类** (`include/llm.hpp`, `src/llm.cpp`): 主要的 LLM 实现，处理模型加载、分词和推理
2. **Tokenizer** (`include/tokenizer.hpp`, `src/tokenizer.cpp`): 处理多种分词器类型 (SentencePiece, TikToken, BERT, HuggingFace)
3. **LlmConfig** (`src/llmconfig.hpp`): 使用 JSON 文件进行配置管理
4. **DiskEmbedding**: 基于磁盘的高效嵌入实现，节省内存

### 主要特性
- 支持多种 LLM 架构 (Qwen, Llama, ChatGLM 等)
- 多模态支持 (视觉和音频)
- KV 缓存管理，实现高效推理
- 大模型的内存映射支持
- LoRA 适配器支持
- 量化选项

### 演示应用程序
1. **cli_demo** (`demo/cli_demo.cpp`): 用于与模型聊天的命令行界面
2. **web_demo** (`demo/web_demo.cpp`): 使用 HTTP 服务器的基于 Web 的界面
3. **embedding_demo** (`demo/embedding_demo.cpp`): 文本嵌入生成
4. **tokenizer_demo** (`demo/tokenizer_demo.cpp`): 分词器测试工具

### 平台支持
- 移动端: Android (Java/Kotlin) 和 iOS (Swift) 应用程序
- 桌面端: Linux、macOS 和 Windows 的 CLI 和 Web 接口
- Python: Python 绑定，便于集成

## 测试
使用模型配置文件运行特定演示来测试功能。可以使用 CLI 演示配合提示文件进行基准测试。