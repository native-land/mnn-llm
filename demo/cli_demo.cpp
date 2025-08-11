//
//  cli_demo.cpp
//  MNN大语言模型命令行演示程序
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "llm.hpp"
#include <fstream>
#include <stdlib.h>

using namespace MNN::Transformer;

/**
 * @brief 运行基准测试，评估模型性能
 * 
 * 该函数从指定文件读取多个提示(prompt)，然后使用模型生成响应，
 * 并统计处理时间、token数量等性能指标。
 * 
 * @param llm 指向大语言模型对象的指针
 * @param prompt_file 包含测试提示的文件路径
 */
void benchmark(Llm* llm, std::string prompt_file) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    
    // 打开提示文件
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;  // 存储所有有效提示
    std::string prompt;                // 临时存储每一行提示
    
    // 逐行读取提示文件
    while (std::getline(prompt_fs, prompt)) {
        // 忽略以'#'开头的注释行
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        
        // 将字符串中的 "\\n" 替换为真正的换行符 '\n'
        std::string::size_type pos = 0;
        while ((pos = prompt.find("\\n", pos)) != std::string::npos) {
            prompt.replace(pos, 2, "\n");
            pos += 1;
        }
        
        // 将处理后的提示添加到提示列表
        prompts.push_back(prompt);
    }
    
    // 性能统计变量初始化
    int prompt_len = 0;        // 所有提示的总token数
    int decode_len = 0;        // 所有生成文本的总token数
    int64_t vision_time = 0;   // 视觉处理总时间(微秒)
    int64_t audio_time = 0;    // 音频处理总时间(微秒)
    int64_t prefill_time = 0;  // 预填充阶段总时间(微秒)
    int64_t decode_time = 0;   // 解码生成阶段总时间(微秒)
    
    // 获取模型状态引用，用于收集性能数据
    auto& state = llm->getState();
    
    // 遍历所有提示并进行处理
    for (int i = 0; i < prompts.size(); i++) {
        const auto& prompt = prompts[i];
        
        // 再次检查是否为注释行（冗余检查）
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        
        // 根据条件选择不同的处理方式
        if (0) {
            // 方式1：流式输出（当前被禁用）
            llm->response(prompt, &std::cout, nullptr, 0);
            // 持续生成直到结束或达到128个token
            while (!llm->stoped() && state.gen_seq_len_ < 128) {
                llm->generate(1);
            }
        } else {
            // 方式2：一次性处理完整提示
            llm->response(prompt);
        }
        
        // 累加各项性能统计数据
        prompt_len += state.prompt_len_;     // 累加提示token数
        decode_len += state.gen_seq_len_;    // 累加生成token数
        vision_time += state.vision_us_;     // 累加视觉处理时间
        audio_time += state.audio_us_;       // 累加音频处理时间
        prefill_time += state.prefill_us_;   // 累加预填充时间
        decode_time += state.decode_us_;     // 累加解码时间
    }
    
    // 将时间从微秒转换为秒
    float vision_s = vision_time / 1e6;
    float audio_s = audio_time / 1e6;
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    
    // 输出性能统计结果
    printf("\n#################################\n");
    printf("prompt tokens num = %d\n", prompt_len);        // 提示token总数
    printf("decode tokens num = %d\n", decode_len);        // 生成token总数
    printf(" vision time = %.2f s\n", vision_s);          // 视觉处理时间
    printf("  audio time = %.2f s\n", audio_s);           // 音频处理时间
    printf("prefill time = %.2f s\n", prefill_s);         // 预填充时间
    printf(" decode time = %.2f s\n", decode_s);          // 解码时间
    printf("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);  // 预填充速度(tokens/秒)
    printf(" decode speed = %.2f tok/s\n", decode_len / decode_s);   // 解码速度(tokens/秒)
    printf("##################################\n");
}

/**
 * @brief 主函数，程序入口点
 * 
 * 程序支持两种运行模式：
 * 1. 交互模式：只提供模型路径，启动交互式聊天
 * 2. 基准测试模式：提供模型路径和提示文件，运行性能基准测试
 * 
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return int 程序退出码
 */
int main(int argc, const char* argv[]) {
    // 检查命令行参数数量，至少需要提供模型路径
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " model_dir <prompt.txt>" << std::endl;
        return 0;
    }
    
    // 获取模型路径参数
    std::string model_dir = argv[1];
    std::cout << "model path is " << model_dir << std::endl;
    
    // 创建并加载大语言模型
    std::unique_ptr<Llm> llm(Llm::createLLM(model_dir));
    llm->load();
    
    // 如果没有提供提示文件，则启动交互式聊天模式
    if (argc < 3) {
        llm->chat();
    }
    
    // 如果提供了提示文件，则运行基准测试
    std::string prompt_file = argv[2];
    benchmark(llm.get(), prompt_file);
    
    return 0;
}
