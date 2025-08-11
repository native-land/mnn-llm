# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains mnn-llm, a project that enables running large language models (LLMs) using the MNN inference engine. The project supports multiple platforms including Linux, macOS, Windows, Android, and iOS. It provides implementations for command-line, web, Android, and iOS interfaces.

The project has been merged into the main MNN repository at https://github.com/alibaba/MNN/tree/master/transformers/llm.

## Build Commands

### Local Compilation
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

### Key Build Options
- `BUILD_FOR_ANDROID`: Compile for Android devices
- `LLM_SUPPORT_VISION`: Enable vision processing capabilities
- `DUMP_PROFILE_INFO`: Dump performance data after each conversation
- Backend options: `-DMNN_CUDA=ON`, `-DMNN_OPENCL=ON`, `-DMNN_METAL=ON`

## Execution Commands

### Linux/macOS
```bash
# CLI demo
./cli_demo ./Qwen2-1.5B-Instruct-MNN/config.json

# Web UI demo
./web_demo ./Qwen2-1.5B-Instruct-MNN/config.json ../web
```

### Windows
```bash
# CLI demo
.\Debug\cli_demo.exe ./Qwen2-1.5B-Instruct-MNN/config.json

# Web UI demo
.\Debug\web_demo.exe ./Qwen2-1.5B-Instruct-MNN/config.json ../web
```

## Code Architecture

### Core Components
1. **Llm class** (`include/llm.hpp`, `src/llm.cpp`): Main LLM implementation handling model loading, tokenization, and inference
2. **Tokenizer** (`include/tokenizer.hpp`, `src/tokenizer.cpp`): Handles multiple tokenizer types (SentencePiece, TikToken, BERT, HuggingFace)
3. **LlmConfig** (`src/llmconfig.hpp`): Configuration management using JSON files
4. **DiskEmbedding**: Efficient disk-based embedding implementation to save memory

### Key Features
- Support for multiple LLM architectures (Qwen, Llama, ChatGLM, etc.)
- Multi-modal support (vision and audio)
- KV cache management for efficient inference
- Memory mapping support for large models
- LoRA adapter support
- Quantization options

### Demo Applications
1. **cli_demo** (`demo/cli_demo.cpp`): Command-line interface for chatting with the model
2. **web_demo** (`demo/web_demo.cpp`): Web-based interface using HTTP server
3. **embedding_demo** (`demo/embedding_demo.cpp`): Text embedding generation
4. **tokenizer_demo** (`demo/tokenizer_demo.cpp`): Tokenizer testing utility

### Platform Support
- Mobile: Android (Java/Kotlin) and iOS (Swift) applications
- Desktop: CLI and web interfaces for Linux, macOS, and Windows
- Python: Python bindings for easy integration

## Testing
Run specific demos with model configuration files to test functionality. Benchmarking can be performed using prompt files with the CLI demo.