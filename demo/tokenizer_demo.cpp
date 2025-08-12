//
//  tokenizer_demo.cpp
//
//  Created by MNN on 2024/01/12.
//  ZhaodeWang
//

// 包含分词器头文件，用于创建和使用分词器对象
#include "tokenizer.hpp"
// 包含文件流头文件，用于读取提示文件内容
#include <fstream>

// 使用MNN::Transformer命名空间，简化代码中类的使用
using namespace MNN::Transformer;

// 分词器演示程序主函数
// argc: 命令行参数的数量
// argv: 命令行参数的数组，argv[0]通常是程序名，argv[1]及以后是传递的参数
int main(int argc, const char* argv[])
{
    // 打印程序标题
    std::cout << "MNN Tokenizer Demo" << std::endl;
    // 打印所有命令行参数信息（调试用）
    std::cout << "第1个参数：" << argv[0] << std::endl;
    std::cout << "第2个参数：" << argv[1] << std::endl;
    std::cout << "第3个参数：" << argv[2] << std::endl;

    // 检查命令行参数是否足够（至少需要程序名、分词器文件路径、提示文件路径）
    if (argc < 3)
    {
        // 参数不足时，提示正确的使用方法
        std::cout << "使用方法: " << argv[0] << " tokenizer.txt prompt.txt" << std::endl;
        return 0; // 程序正常退出
    }
    else
    {
        std::cout << "参数校验通过" << std::endl;
    }

    // 获取分词器模型路径和提示文件路径
    // argv[1]是分词器配置文件路径，argv[2]是包含测试提示的文件路径
    std::string tokenizer_path = argv[1];
    std::string prompt_file = argv[2];

    // 创建分词器实例
    // 使用智能指针管理分词器对象，避免内存泄漏
    // Tokenizer::createTokenizer是工厂方法，根据配置文件创建具体类型的分词器
    std::unique_ptr<Tokenizer> tokenizer(Tokenizer::createTokenizer(tokenizer_path));

    // 打开提示文件
    // 使用ifstream打开文件，如果文件不存在或无法打开会设置错误标志
    std::ifstream prompt_fs(prompt_file);

    // 定义存储提示的向量和临时字符串变量
    std::vector<std::string> prompts;
    std::string prompt;

    // 逐行读取提示文件内容
    // std::getline从文件流中读取一行到prompt变量中，直到文件结束
    while (std::getline(prompt_fs, prompt))
    {
        std::cout << "prompt:" << prompt << std::endl;

        // 以'#'开头的提示行将被忽略（作为注释行处理）
        if (prompt.substr(0, 1) == "#")
        {
            continue; // 跳过当前循环，处理下一行
        }

        // 处理字符串中的换行符转义序列
        // 将文本中的"\\n"替换为实际的换行符'\n'
        std::string::size_type pos = 0;
        while ((pos = prompt.find("\\n", pos)) != std::string::npos)
        {
            // 找到"\\n"时，将其替换为'\n'
            prompt.replace(pos, 2, "\n");
            // 更新位置，避免重复处理
            pos += 1;
        }

        // 构造查询字符串，添加特定的前缀和后缀
        // 这种格式通常用于指令微调的模型
        const std::string query = "\n### Human: " + prompt + "\n### Assistant: ";
        std::cout << query << std::endl;

        // 使用分词器对查询字符串进行编码
        // encode方法将文本转换为token ID序列
        auto tokens = tokenizer->encode(query);

        // 初始化解码字符串，用于存储解码后的结果
        std::string decode_str;

        // 打印编码后的token序列
        printf("编码后的token序列 = [ ");
        // 遍历token序列中的每个token
        for (auto token : tokens)
        {
            // 打印token ID
            printf("%d, ", token);
            // 对每个token进行解码并拼接成完整字符串
            // decode方法将单个token ID转换为对应的文本
            decode_str += tokenizer->decode(token);
        }
        printf("]\n");

        // 打印解码后的字符串
        // 验证编码和解码过程的正确性
        printf("解码后的字符串 = %s\n", decode_str.c_str());
    }

    // 程序正常结束
    return 0;
}
