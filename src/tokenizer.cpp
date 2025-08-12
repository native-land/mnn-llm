//
//  tokenizer.cpp
//
//  Created by MNN on 2023/09/25.
//  ZhaodeWang
//

#include "tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <queue>
#include <functional>
#include <random>
#include <codecvt>
#include <regex>
#include <set>
#include <climits>
#include <cctype>
namespace MNN {
namespace Transformer {

// 用于解码的Base64字符集
static const std::string base64_chars =
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz"
"0123456789+/";

// 检查字符是否为有效的base64字符
static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

// 根据UTF-8字符的第一个字节获取字符的字节长度
// 使用查找表确定字符长度:
// 0xxxxxxx -> 1字节
// 110xxxxx -> 2字节
// 1110xxxx -> 3字节
// 11110xxx -> 4字节
static inline size_t one_char_len(const char *src) {
    return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

// 将base64编码的字符串解码回原始二进制表示
// 此函数实现了标准的base64解码算法:
// Base64编码原理:
// 1. 将每3个字节(24位)的二进制数据分割成4组，每组6位
// 2. 将每组6位映射到64个字符中的一个('A'-'Z','a'-'z','0'-'9','+','/')
// 3. 如果最后一组不足3字节，则用'='填充
//
// 解码过程:
// 1. 每次处理4个base64字符以产生3个输出字节
// 2. 处理末尾的填充字符(=)
// 3. 将base64字符转换为6位值并重新组合成字节
static std::string base64_decode(const std::string& str) {
    int in_len = str.size();           // 输入字符串长度
    int i = 0;                         // 4字符处理组内的索引
    int j = 0;                         // 填充的循环索引
    int in_ = 0;                       // 输入字符串中的索引
    unsigned char char_array_4[4], char_array_3[3];  // 处理用的临时数组
    std::string ret;                   // 结果字符串

    // 主解码循环 - 每次处理4个base64字符
    // 循环条件: 还有字符未处理 且 当前字符不是填充符 且 当前字符是有效的base64字符
    while (in_len-- && ( str[in_] != '=') && is_base64(str[in_])) {
        char_array_4[i++] = str[in_]; in_++;
        // 当收集到4个base64字符时，进行一次解码
        if (i ==4) {
            // 将base64字符转换为6位值
            // 通过在base64_chars中查找字符位置来获取其6位值
            for (i = 0; i <4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }
            // 将4组6位重新组合成3个字节
            // Base64解码的核心位操作:
            // 4个6位值(24位)重新组合成3个8位字节
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
            // 将解码后的字节添加到结果中
            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }
            i = 0;  // 重置为下一组
        }
    }
    // 处理剩余字符(当输入长度不是4的倍数时)
    // 这种情况发生在输入数据长度不是3的倍数时，需要特殊处理
    if (i) {
        // 用零填充
        // 将未填满的数组位置填充为0
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }
        // 转换为6位值
        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }
        // 重新组合剩余字节
        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
        // 仅将有效字节添加到结果中(i-1字节)
        // 只添加实际有效的字节数，避免添加由填充产生的无效字节
        for (j = 0; (j < i - 1); j++) {
            ret.push_back(char_array_3[j]);
        }
    }
    return ret;
}

// 通过就地修改每个字符将字符串转换为小写
static inline void to_lower_case(std::string& str) {
    for (auto &c : str) {
        if (c >= 'A' && c <= 'Z') {
            c = tolower(static_cast<unsigned char>(c));
        }
    }
}

// 工厂方法，根据文件中存储的分词器类型创建分词器实例
// 函数读取分词器文件头以确定要实例化的分词器实现
// 支持的类型: SENTENCEPIECE, TIKTOIKEN, BERT, HUGGINGFACE
Tokenizer* Tokenizer::createTokenizer(const std::string& filename) {
    Tokenizer* tokenizer = nullptr;
    // 检查分词器文件是否存在且可以打开
    std::ifstream tok_file(filename);
    if (!tok_file.good()) {
        printf("失败: 无法从 %s 加载分词器。\n", filename.c_str());
        return tokenizer;
    }
    // 读取第一行，其中包含魔数和分词器类型
    std::string line;
    std::getline(tok_file, line);
    std::istringstream line_str(line);
    int magic_number, tokenizer_type;
    line_str >> magic_number;
    // 验证魔数以确保文件格式兼容性
    if (magic_number != MAGIC_NUMBER) {
        printf("失败: %s 的魔数错误。\n", filename.c_str());
        return tokenizer;
    }
    // 从头部提取分词器类型
    line_str >> tokenizer_type;
    printf("tokenizer_type = %d\n", tokenizer_type);
    // 根据类型实例化相应的分词器
    switch (tokenizer_type)
    {
        case SENTENCEPIECE:
            tokenizer = new Sentencepiece();
            break;
        case TIKTOIKEN:
            tokenizer = new Tiktoken();
            break;
        case BERT:
            tokenizer = new BertTokenizer();
            break;
        case HUGGINGFACE:
            tokenizer = new HuggingfaceTokenizer();
            break;
        default:
            return tokenizer;
    }
    // 从文件加载特殊令牌(特殊令牌、停止令牌和前缀令牌)
    tokenizer->load_special(tok_file);
    // 加载特定于分词器类型的词汇表数据
    tokenizer->load_vocab(tok_file);
    tok_file.close();
    return tokenizer;
}

// 检查令牌ID是否在停止令牌列表中
// 停止令牌用于确定语言模型何时停止生成
bool Tokenizer::is_stop(int token) {
    return std::find(stop_tokens_.begin(), stop_tokens_.end(), token) != stop_tokens_.end();
}

// 检查令牌ID是否在特殊令牌列表中
// 特殊令牌具有特定含义(例如，<BOS>, <EOS>, <PAD>)并以不同方式处理
bool Tokenizer::is_special(int token) {
    return std::find(special_tokens_.begin(), special_tokens_.end(), token) != special_tokens_.end();
}

// 从分词器文件加载特殊令牌
// 这些包括特殊令牌、停止令牌和前缀令牌
// 格式: 第一行有计数，第二行有实际的令牌ID
void Tokenizer::load_special(std::ifstream& tok_file) {
    std::string line;
    std::getline(tok_file, line);
    std::istringstream line_str(line);
    int special_num, stop_num, prefix_num;
    // 读取每种令牌类型的计数
    line_str >> special_num >> stop_num >> prefix_num;
    std::getline(tok_file, line);
    std::istringstream specail_line(line);
    // 如果存在特殊令牌则加载
    if (special_num) {
        // 加载特殊令牌
        special_tokens_.resize(special_num);
        for (int i = 0; i < special_num; i++) {
            specail_line >> special_tokens_[i];
        }
    }
    // 如果存在停止令牌则加载
    if (stop_num) {
        // 加载停止令牌
        stop_tokens_.resize(stop_num);
        for (int i = 0; i < stop_num; i++) {
            specail_line >> stop_tokens_[i];
        }
    }
    // 如果存在前缀令牌则加载
    if (prefix_num) {
        // 加载前缀令牌
        prefix_tokens_.resize(prefix_num);
        for (int i = 0; i < prefix_num; i++) {
            specail_line >> prefix_tokens_[i];
        }
    }
}

// 主编码函数，将文本转换为令牌ID
// 此函数在调用特定分词器实现之前处理特殊令牌和前缀
// 处理流程:
// 1. 从预定义的前缀令牌开始(如BOS令牌)
// 2. 如果存在特殊令牌(如<EOS>, <PAD>等):
//    a. 扫描输入文本查找特殊令牌
//    b. 对特殊令牌之间的普通文本调用具体分词器的encode方法
//    c. 将特殊令牌直接添加到结果中
// 3. 如果没有特殊令牌，直接对整个字符串调用具体分词器的encode方法
std::vector<int> Tokenizer::encode(const std::string& str) {
    // 从前缀令牌开始(例如，BOS令牌)
    std::vector<int> ids = prefix_tokens_;
    // 如果词汇表中存在特殊令牌则处理
    if (!special_tokens_.empty()) {
        std::string text = str;
        size_t start = 0;
        // 扫描文本以查找和处理特殊令牌
        for (size_t i = 0; i < text.length(); ++i) {
            // 检查每个特殊令牌以查看它是否在当前位置匹配
            for (auto special_id : special_tokens_) {
                const auto& token = decode(special_id);
                if (token.empty()) continue;
                // 如果特殊令牌在当前位置匹配
                if (i + token.length() <= text.length() && text.substr(i, token.length()) == token) {
                    // 对特殊令牌之前的文本进行编码
                    if (i > start) {
                        encode(text.substr(start, i - start), ids);
                    }
                    // 将特殊令牌ID添加到结果中
                    ids.push_back(special_id);
                    // 将起始位置移过特殊令牌
                    start = i + token.length();
                    i = start - 1;
                    break;
                }
            }
        }
        // 对最后一个特殊令牌之后的任何剩余文本进行编码
        if (start < text.length()) {
            encode(text.substr(start), ids);
        }
    } else {
        // 没有特殊令牌，对整个字符串进行编码
        encode(str, ids);
    }
    return ids;
}

// 从分词器文件加载SentencePiece分词器的词汇表数据
// 词汇表包含带有分数和类型的片段
// 文件格式: 
// - 第一行: 词汇表大小
// - 后续每行: base64编码的令牌、分数和片段类型
// 
// SentencePiece词汇表特点:
// 1. 使用Base64编码存储令牌，支持任意字节序列
// 2. 每个片段都有分数，用于BPE合并时的优先级判断
// 3. 片段分为不同类型(NORMAL, CONTROL, UNKNOWN等)
bool Sentencepiece::load_vocab(std::ifstream& tok_file) {
    std::string line, token;
    // 从第一行读取词汇表大小
    // 第一行包含一个整数，表示词汇表中片段的总数
    std::getline(tok_file, line);
    int vocab_len = std::stoi(line);
    float score;
    int type;
    // 调整句子片段向量大小以容纳所有词汇项目
    // sentence_pieces_存储所有片段的完整信息(令牌、分数、类型)
    sentence_pieces_.resize(vocab_len);
    // 加载每个词汇项目
    // 逐行读取词汇表数据，每行包含一个片段的信息
    for (int index = 0; index < vocab_len; index++) {
        std::getline(tok_file, line);
        std::istringstream line_str(line);
        // 从行中解析令牌、分数和类型
        // 格式: base64_token score type
        line_str >> token >> score >> type;
        // 将base64编码的令牌解码回原始字符串
        // SentencePiece使用Base64编码来支持任意字节序列的存储
        token = base64_decode(token);
        // 将整数类型转换为PieceType枚举
        // PieceType定义了片段的类型，如NORMAL, CONTROL, UNKNOWN等
        auto piece_type = static_cast<PieceType>(type);
        // 创建带有令牌、分数和类型的SentencePiece对象
        SentencePiece piece = {token, score, piece_type};
        sentence_pieces_[index] = std::move(piece);
        // 根据类型对片段进行分类
        // 不同类型的片段存储在不同的数据结构中，便于快速查找
        if (piece_type == PieceType::NORMAL) {
            // 正常片段存储在pieces_映射中以便快速查找
            // NORMAL片段是普通的词汇表条目，用于常规的分词操作
            pieces_.insert({token, index});
        } else {
            // 特殊片段(控制、未知等)存储在reserved_id_map_中
            // 控制片段包括<BOS>, <EOS>等特殊标记
            // 未知片段是用于处理词汇表外词的特殊标记
            reserved_id_map_.insert({token, index});
            if (piece_type == PieceType::UNKNOWN) {
                // 跟踪未知令牌ID
                // UNKNOWN片段用于处理不在词汇表中的词
                unk_id_ = index;
            }
        }
    }
    return true;
}

// 将片段(令牌字符串)转换为其对应的ID
// 首先检查保留令牌，然后检查正常令牌，最后回退到未知令牌
int Sentencepiece::piece_to_id(const std::string& piece) const {
    // 检查片段是否在保留令牌中(控制、未知等)
    auto it = reserved_id_map_.find(piece);
    if (it != reserved_id_map_.end()) {
        return it->second;
    }
    // 检查片段是否在正常令牌中
    auto it2 = pieces_.find(piece);
    if (it2 != pieces_.end()) {
        return it2->second;
    }
    // 如果未找到片段则返回未知令牌ID
    return unk_id_;
}

// 将字节值转换为其对应的片段表示
// 格式: <0xXX> 其中XX是字节的十六进制表示
std::string Sentencepiece::byte_to_piece(unsigned char c) const {
    const int len = ::snprintf(nullptr, 0, "<0x%02X>", c);
    std::string s;
    s.resize(len);
    ::snprintf(&s[0], s.size() + 1, "<0x%02X>", c);
    return s;
}

// 在标准化字符串上执行BPE(字节对编码)
// 这实现了SentencePiece算法的核心文本分词功能
// 算法步骤:
// 1. 将输入文本分割成UTF-8字符级别的初始符号
// 2. 计算所有相邻符号对的合并分数并放入优先队列
// 3. 重复选择分数最高的符号对进行合并，直到无法合并为止
// 4. 处理未使用的片段，将其递归分解为更小的已知片段
// 参考: https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc
Sentencepiece::EncodeResult Sentencepiece::bpe_encode(string_view_ normalized, float alpha) {
    // 表示具有合并分数的相邻符号对的结构
    // 在BPE过程中，我们会不断合并相邻的符号对，这个结构用于跟踪哪些对可以合并以及合并的优先级
    struct SymbolPair {
        int left;     // 此对的左符号索引
        int right;    // 此对的右符号索引
        float score;  // 此对的合并分数。分数越高表示优先级越高，越应该被合并。
        size_t size;  // 合并后片段的长度(用于验证合并的有效性)
    };

    // 优先队列中符号对优先级的比较器
    // 用于确定哪个符号对应该优先合并:
    // 1. 分数越高优先级越高
    // 2. 如果分数相同，左侧索引较小的优先级更高
    class SymbolPairComparator {
    public:
        const bool operator()(SymbolPair *h1, SymbolPair *h2) {
            return (h1->score < h2->score || (h1->score == h2->score && h1->left > h2->left));
        }
    };

    // 表示分割过程中符号的结构
    // 使用双向链表结构来表示符号序列，便于在合并过程中快速更新相邻关系
    struct Symbol {
        int prev;     // 此符号的前一个符号索引。对于序列开始，值为-1。
        int next;     // 此符号的下一个符号索引。对于序列结束，值为-1。
        bool freeze = false;  // 此符号是否被冻结(不会参与合并)。
        string_view_ piece;   // 此符号的实际文本内容
    };

    // 按合并分数管理符号对的优先队列
    // 优先队列确保我们总是处理当前分数最高的符号对
    using Agenda = std::priority_queue<SymbolPair *, std::vector<SymbolPair *>, SymbolPairComparator>;
    Agenda agenda;
    // 在分割过程中保存所有符号的向量
    // 这个向量表示当前的符号序列，在BPE过程中会动态更新
    std::vector<Symbol> symbols;
    symbols.reserve(normalized.size());
    // 用于未使用片段重新分割的反向合并规则
    // 当一个片段被标记为"未使用"时，这个映射表记录了它是如何由两个子片段合并而来的
    // 这样我们可以在最后阶段将未使用的片段递归分解为已知的子片段
    std::unordered_map<string_view_, std::pair<string_view_, string_view_>> rev_merge;
    // 符号对对象的存储以避免重复分配
    // 用于存储在BPE过程中创建的SymbolPair对象，避免频繁的内存分配
    std::vector<std::unique_ptr<SymbolPair>> symbol_pair_holder;
    
    // 检查合并两个相邻符号是否会创建有效片段的函数
    // 如果合并后的片段存在于词汇表中，则将该对添加到议程中并附带其分数
    auto MaybeAddNewSymbolPair = [this, &symbol_pair_holder, &symbols, &agenda, &rev_merge](int left, int right) {
        // 如果任一符号无效或已冻结则跳过
        // 无效符号指索引为-1的符号，冻结符号是不应该再参与合并的符号
        if (left == -1 || right == -1 || symbols[left].freeze || symbols[right].freeze) {
            return;
        }
        // 从两个符号创建合并的片段
        // 将左右两个符号的文本内容连接起来形成新的片段
        const string_view_ piece(symbols[left].piece.data(), symbols[left].piece.size() + symbols[right].piece.size());
        std::string piece_str(piece.to_string());
        // 检查合并的片段是否存在于我们的词汇表中
        // 只有存在于词汇表中的片段才能进行合并
        const auto it = pieces_.find(piece_str);
        if (it == pieces_.end()) {
            return;
        }
        // 创建新的符号对并将其添加到议程中
        symbol_pair_holder.emplace_back(new SymbolPair);
        auto *h = symbol_pair_holder.back().get();
        h->left = left;
        h->right = right;
        h->score = get_score(it->second);  // 获取此片段的分数，分数来自训练时的统计信息
        h->size = piece.size();
        agenda.push(h);

        // 存储未使用片段重新分割的反向合并信息
        // 如果这个片段被标记为"未使用"，记录它是如何由两个子片段合并而来的
        if (is_unused(it->second)) {
            rev_merge[piece] = std::make_pair(symbols[left].piece, symbols[right].piece);
        }
    };
    
    // 将输入分割成初始字符级符号
    // BPE的起点是将输入文本按UTF-8字符边界分割成初始符号
    int index = 0;
    while (!normalized.empty()) {
        Symbol s;
        // 确定当前UTF-8字符的长度
        // 使用one_char_len函数根据UTF-8编码规则确定字符的字节长度
        int mblen = std::min<int>(normalized.size(), one_char_len(normalized.data()));
        s.piece = string_view_(normalized.data(), mblen);
        // 为相邻符号设置链表指针
        // 构建双向链表结构，便于在后续合并过程中快速找到相邻符号
        s.prev = index == 0 ? -1 : index - 1;
        normalized.remove_prefix(mblen);
        s.next = normalized.empty() ? -1 : index + 1;
        ++index;
        symbols.emplace_back(s);
    }

    // 如果未创建符号则返回空结果
    // 处理空输入的边界情况
    if (symbols.empty()) {
        return {};
    }
    
    // 使用所有相邻符号对初始化议程
    // 将初始符号序列中所有相邻的符号对加入优先队列
    for (size_t i = 1; i < symbols.size(); ++i) {
        MaybeAddNewSymbolPair(i - 1, i);
    }

    // BPE-dropout正则化机制 (https://arxiv.org/pdf/1910.13267.pdf)
    // 这是一种正则化技术，以一定概率跳过某些合并操作，增加模型的鲁棒性
    std::mt19937 rand_gen;
    auto skip_merge = [&]() {
        // 以alpha概率跳过合并(0.0 = 不dropout, 1.0 = 总是dropout)
        if (alpha <= 0.0) return false;
        if (alpha >= 1.0) return true;
        std::uniform_real_distribution<> gen(0.0, 1.0);
        return gen(rand_gen) < alpha;
    };

    // 主BPE合并循环
    // 重复选择分数最高的符号对进行合并，直到无法合并为止
    while (!agenda.empty()) {
        // 从议程中获取分数最高的对
        SymbolPair *top = agenda.top();
        agenda.pop();

        // 如果此对不再有效(符号已被合并)则跳过
        // 在合并过程中，某些符号可能已经被合并，需要检查当前对是否仍然有效
        if (symbols[top->left].piece.empty() || symbols[top->right].piece.empty() ||
            symbols[top->left].piece.size() + symbols[top->right].piece.size() != top->size) {
            continue;
        }

        // 如果启用则应用BPE-dropout
        // 根据概率决定是否跳过这次合并
        if (skip_merge()) continue;
        
        // 将两个符号合并为一个
        // 更新左符号的内容为合并后的内容
        symbols[top->left].piece = string_view_(
                                                symbols[top->left].piece.data(),
                                                symbols[top->left].piece.size() + symbols[top->right].piece.size());

        // 更新链表指针以移除右侧符号
        // 从链表中移除右符号，将其与左符号合并
        symbols[top->left].next = symbols[top->right].next;
        if (symbols[top->right].next >= 0) {
            symbols[symbols[top->right].next].prev = top->left;
        }
        symbols[top->right].piece = string_view_("");

        // 添加由此合并操作创建的新潜在合并
        // 合并操作可能创造了新的相邻符号对，需要将它们加入议程
        MaybeAddNewSymbolPair(symbols[top->left].prev, top->left);
        MaybeAddNewSymbolPair(top->left, symbols[top->left].next);
    }

    // 将未使用片段递归重新分割成更小的已知片段的函数
    // 对于标记为"未使用"的片段，递归地将其分解为更小的已知片段
    std::function<void(string_view_, EncodeResult*)> resegment;
    resegment = [this, &resegment, &rev_merge](string_view_ w, EncodeResult *output) -> void {
        std::string w_str(w.to_string());
        const int id = piece_to_id(w_str);
        // 如果片段有效且未使用，则将其添加到输出中
        // 有效的片段是指在词汇表中的片段，未使用的片段需要进一步分解
        if (id == -1 || !is_unused(id)) {
            output->emplace_back(w, id);
            return;
        }
        // 对于未使用片段，查找如何分解它们
        // 使用之前记录的反向合并信息进行分解
        const auto p = rev_merge.find(w);
        if (p == rev_merge.end()) {
            // 此块永远不会被调用，因为`rev_merge`存储了所有未使用ID的
            // 重新分割信息。
            output->emplace_back(w, id);
            return;
        }
        // 递归重新分割组成部分
        // 递归地处理分解后的两个子片段
        resegment(p->second.first, output);
        resegment(p->second.second, output);
    };
    
    // 通过遍历符号链表生成最终编码结果
    // 遍历最终的符号链表，对每个符号调用resegment函数得到最终的编码结果
    EncodeResult output;
    for (int index = 0; index != -1; index = symbols[index].next) {
        resegment(symbols[index].piece, &output);
    }
    return output;
}

// 使用SentencePiece算法将字符串编码为令牌ID
// 如果启用则使用字节级回退处理未知令牌
// SentencePiece编码完整流程:
// 1. 调用bpe_encode进行BPE分词，得到片段-ID对
// 2. 遍历所有片段，将它们转换为对应的ID
// 3. 对于未知片段(不在词汇表中)，如果启用字节级回退则将其分解为字节级表示
void Sentencepiece::encode(const std::string& str, std::vector<int>& ids) {
    // 执行BPE编码以获取片段-字符串对
    // bpe_encode是SentencePiece的核心分词函数，返回分词结果
    auto result = bpe_encode(str);
    size_t consumed = 0;
    // 将每个片段转换为其对应的ID
    // 遍历BPE分词结果，将每个片段转换为对应的词汇表ID
    for (const auto &p : result) {
        const string_view_ w = p.first;   // 片段的字符串内容
        const int id = p.second;          // 片段对应的词汇表ID
        const bool is_unk = (id == unk_id_);  // 判断是否为未知令牌
        // 如果启用且存在未知令牌则使用字节级回退处理
        // 字节级回退是一种处理未知词的策略:
        // 将未知词分解为UTF-8字节，每个字节用特殊标记表示
        if (is_unk && byte_fall_back_) {
            // 将未知片段分解为UTF-8字节
            // 对于每个字节，创建其对应的特殊表示并查找ID
            for (int i = 0; i < w.size(); ++i) {
                // 创建字节片段表示
                // byte_to_piece将字节转换为"<0xXX>"格式的字符串
                const char b = w[i];
                const auto piece = byte_to_piece(b);
                // 查找字节片段对应的ID
                auto sp_id = piece_to_id(piece);
                ids.push_back(sp_id);
            }
        } else {
            // 添加正常令牌ID
            // 对于已知片段，直接添加其ID
            ids.push_back(id);
        }
    }
}

// 将令牌ID解码回其字符串表示
// 处理SentencePiece使用的特殊下划线字符
std::string Sentencepiece::decode(int id) {
    // 获取此ID的片段字符串
    auto piece = sentence_pieces_[id].piece;
    // 用空格替换SentencePiece的特殊下划线字符
    int pos = piece.find("▁");
    if (pos != -1) {
        piece.replace(pos, pos + 3, " ");
    }
    return piece;
}

// 通过ID获取片段的分数
// 分数用于确定BPE中的合并优先级
float Sentencepiece::get_score(int id) const {
    return sentence_pieces_[id].score;
}

// 检查片段是否标记为未使用
bool Sentencepiece::is_unused(int id) const {
    return sentence_pieces_[id].type == PieceType::UNUSED;
}

// 检查片段是否为控制令牌
bool Sentencepiece::is_control(int id) const {
    return sentence_pieces_[id].type == PieceType::CONTROL;
}

// 从分词器文件加载Tiktoken分词器的词汇表
// 词汇表包含用于编码和解码的令牌到ID的映射
// 格式: 第一行是词汇表大小，后面是base64编码的令牌
bool Tiktoken::load_vocab(std::ifstream& tok_file) {
    std::string line;
    // 从第一行读取词汇表大小
    std::getline(tok_file, line);
    int vocab_len = std::stoi(line);
    // 调整解码器向量大小以容纳所有令牌
    decoder_.resize(vocab_len);
    // 加载每个令牌并构建编码器/解码器映射
    for (int i = 0; i < vocab_len; i++) {
        std::getline(tok_file, line);
        // 将base64编码的令牌解码回原始字符串
        auto token = base64_decode(line);
        // 构建双向映射:
        // encoder_: 令牌字符串 -> ID (用于编码)
        // decoder_: ID -> 令牌字符串 (用于解码)
        encoder_.insert({token, i});
        decoder_[i] = token;
    }
    return true;
}

// 使用贪心最长匹配算法将字符串编码为令牌ID
// 这是Tiktoken编码的核心方法 - 在每个位置找到最长匹配的令牌
// 算法原理:
// 1. 从左到右扫描输入字符串
// 2. 在每个位置，尝试匹配最长的已知令牌
// 3. 贪心策略: 总是选择在当前位置能匹配到的最长令牌
// 4. 将匹配到的令牌转换为对应的ID并添加到结果中
// 5. 移动到下一个未处理的位置，重复上述过程
void Tiktoken::encode(const std::string& str, std::vector<int>& ids) {
    // 空字符串的早期返回
    if (str.empty()) {
        return;
    }
    size_t i = 0;
    // 从左到右处理字符串
    while (i < str.size()) {
        // bool found_pair = false;
        // 尝试在当前位置匹配最长可能的符号
        size_t longest_match_len = 0;
        std::string longest_match;

        // 检查递减长度的子字符串以找到最长匹配
        // 这确保我们总是使用最长可能的令牌
        // 从最长可能的子串开始，逐渐缩短，直到找到匹配的令牌
        for (size_t len = str.size() - i; len > 0; --len) {
            std::string token = str.substr(i, len);
            auto it = encoder_.find(token);
            // 如果令牌存在于词汇表中且比之前的匹配更长
            if (it != encoder_.end()) {
                if (len > longest_match_len) {
                    longest_match_len = len;
                    longest_match = it->first;
                }
            }
        }

        // 如果找到则将匹配的令牌ID添加到结果中
        if (!longest_match.empty()) {
            ids.push_back(encoder_.at(longest_match));
            i += longest_match_len;  // 按匹配令牌的长度前进
        } else {
            // 使用正确训练的分词器时不应发生这种情况
            // 表示分词器或输入文本存在问题
            std::cerr << "错误: 在位置 " << i << " 开始的序列未找到编码" << std::endl;
            return;
        }
    }
}

// 将令牌ID解码回其字符串表示
// 在解码器数组中简单查找
std::string Tiktoken::decode(int id) {
    // 边界检查以防止数组访问错误
    if (id >= decoder_.size()) {
        return "";
    }
    return decoder_[id];
}

// 对单个令牌应用WordPiece分词
// 这是BERT分词的核心算法，通过将词汇表外的词分解为子词单元来处理
// WordPiece算法原理:
// 1. 首先尝试将整个令牌作为一个整体进行匹配
// 2. 如果整个令牌不在词汇表中，则将其分解为子词片段
// 3. 分解策略: 贪心地在每个位置匹配最长的已知片段
// 4. 连续片段使用"##"前缀标识，表示这不是一个完整单词的开始
// 5. 如果无法匹配任何片段，则使用[UNK]令牌
std::vector<int> BertTokenizer::word_piece(const std::string& token) {
    // 首先检查整个令牌是否存在于词汇表中
    // 这是最优情况，可以直接匹配整个令牌
    auto it = encoder_.find(token);
    if (it != encoder_.end()) {
        return {it->second};
    }
    
    // 如果未找到，则分解为子词片段
    // 使用贪心算法将令牌分解为多个子词片段
    std::vector<int> ids;
    std::string current = token;
    while (!current.empty()) {
        int match_id = -1;      // 匹配片段的ID
        size_t match_pos = 0;   // 匹配结束的位置
        
        // 尝试在当前字符串开头匹配最长可能的片段
        // 从最长可能的子串开始，逐渐缩短，直到找到匹配的片段
        for (int len = current.size(); len > 0; --len) {
            std::string candidate = current.substr(0, len);
            // 为连续片段添加"##"前缀(除了第一个片段)
            // "##"前缀表示这不是一个完整单词的开始，而是单词的 continuation
            if (!ids.empty()) {
                candidate = "##" + candidate;
            }
            auto it = encoder_.find(candidate);
            // 如果片段存在于词汇表中，则使用它
            if (it != encoder_.end()) {
                match_id = it->second;
                match_pos = len;
                break;
            }
        }
        
        // 处理未知令牌 - 使用[UNK]令牌(ID 100)
        // 如果在当前位置无法匹配任何已知片段，则将整个令牌标记为未知
        if (match_id == -1) {
            ids.push_back(100);  // [UNK]令牌ID
            break;
        }
        
        // 将匹配的片段ID添加到结果中
        ids.push_back(match_id);
        // 移动到字符串的剩余部分
        // 非第一个词，添加##前缀
        current = current.substr(match_pos);
    }
    return ids;
}

// 使用BERT分词方法将字符串编码为令牌ID
// BERT分词完整流程:
// 1. 预分词: 将文本分割为基本令牌(单词、标点符号等)
// 2. WordPiece分词: 对每个基本令牌应用WordPiece算法进一步细分
// BERT分词特点:
// 1. 将文本转换为小写(原始BERT是大小写不敏感的)
// 2. 将连续的字母数字字符视为一个令牌
// 3. 将标点符号单独作为令牌处理
// 4. 使用WordPiece处理词汇表外的词
void BertTokenizer::encode(const std::string& str, std::vector<int>& ids) {
    // 第一遍: 将文本预分词为基本令牌
    // 这是BERT分词的第一步，将输入文本分割成粗粒度的令牌
    std::vector<std::string> tokens;
    std::string current_token;
    size_t i = 0;
    
    // 处理每个字符以构建基本令牌
    // 遍历输入字符串的每个字符，根据字符类型进行分组
    while (i < str.size()) {
        current_token.clear();
        unsigned char c = static_cast<unsigned char>(str[i]);
        
        // 处理多字节UTF-8字符(简化处理3字节字符)
        // 检测UTF-8多字节字符(这里简化处理，只检测3字节字符)
        if ((c & 0x80) != 0) {
            unsigned char mask = 0xE0; // 1110 0000 表示3字节字符
            if ((c & mask) == mask) {
                current_token = str.substr(i, 3);
                i += 3;
            } else {
                ++i;
                continue;
            }
        }
        // 处理连续的字母和数字序列
        // 将连续的字母数字字符分组为一个令牌，并转换为小写
        else if (isalnum(c)) {
            // 将连续的字母数字字符分组并转换为小写
            while (i < str.size() && isalnum(static_cast<unsigned char>(str[i]))) {
                current_token += tolower(str[i]);
                ++i;
            }
        }
        // 处理标点符号和符号
        // 将每个标点符号单独作为一个令牌
        else if (ispunct(c)) {
            current_token = str[i];
            ++i;
        }
        // 处理空格、制表符、回车
        // 跳过空白字符
        else if (isspace(c)) {
            ++i;
            continue;  // 跳过空白字符
        }
        // 处理任何其他单字节字符
        // 将其他字符单独作为一个令牌
        else {
            current_token = str[i];
            ++i;
        }
        
        // 将非空令牌添加到列表中
        // 只有非空的令牌才会被添加到结果中
        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }
    }

    // 第二遍: 对每个基本令牌应用WordPiece分词
    // 对预分词得到的每个基本令牌应用WordPiece算法进行细粒度分词
    for (auto token : tokens) {
        // 根据需要将每个令牌细分为子词片段
        // 调用word_piece函数对每个令牌进行WordPiece分词
        for (auto id : word_piece(token)) {
            ids.push_back(id);
        }
    }
}

// 将UTF-8编码的字符串转换为宽字符串(wstring)
// 使用标准库codecvt功能进行UTF-8到UTF-32/UTF-16转换
// 这些函数在HuggingFace分词器中用于处理Unicode字符
// UTF-8到宽字符串的转换:
// 1. UTF-8是一种变长编码，1-4字节表示一个字符
// 2. 宽字符串(wstring)使用固定长度编码(通常是UTF-16或UTF-32)
// 3. 转换过程将UTF-8字节序列映射到对应的宽字符
std::wstring utf8_to_wstring(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.from_bytes(str);
}

// 将宽字符串(wstring)转换为UTF-8编码的字符串
// 使用标准库codecvt功能进行UTF-32/UTF-16到UTF-8转换
// 宽字符串到UTF-8的转换:
// 1. 宽字符(UTF-16/UTF-32)是固定长度编码
// 2. UTF-8是变长编码，1-4字节表示一个字符
// 3. 转换过程将宽字符映射到对应的UTF-8字节序列
std::string wstring_to_utf8(const std::wstring& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.to_bytes(str);
}

// 给定一个UTF-8字符串令牌，使用提供的字节到unicode映射表
// 将每个字节编码为宽字符
// 这是HuggingFace分词器中BPE分词预处理的一部分
void byte_encode_token(const std::string& token,
                       const std::unordered_map<uint8_t, wchar_t>& b2u,
                       std::wstring* result) {
    result->resize(0);
    // 处理令牌中的每个字节
    for (char c : token) {
        // 使用映射表将字节转换为宽字符
        wchar_t wc = b2u.at(uint8_t(c));
        result->push_back(wc);
    }
}

// 为HuggingFace BPE分词器加载词汇表和合并规则
// 这包括令牌词汇表和BPE合并操作
// 文件格式:
// - 第一行: 词汇表大小和合并规则计数
// - 接下来的vocab_len行: 词汇表令牌
// - 接下来的merge_len行: 合并规则(令牌对)
// 
// HuggingFace分词器特点:
// 1. 使用预训练的词汇表和合并规则
// 2. 采用字节级别的预处理，确保所有字节都能正确表示
// 3. BPE合并规则有明确的优先级顺序
bool HuggingfaceTokenizer::load_vocab(std::ifstream& tok_file) {
    std::string line, token;
    // 从第一行读取词汇表和合并规则计数
    // 第一行包含两个整数: 词汇表大小和合并规则数量
    int vocab_len, merge_len;
    std::getline(tok_file, line);
    std::istringstream line_str(line);
    line_str >> vocab_len >> merge_len;
    
    // 将词汇表令牌加载到编码器/解码器映射中
    // 构建双向映射结构，支持快速的编码和解码操作
    decoder_.resize(vocab_len);
    for (int i = 0; i < vocab_len; i++) {
        std::getline(tok_file, line);
        // 构建双向映射:
        // encoder_: 令牌字符串 -> ID (用于编码)
        // decoder_: ID -> 令牌字符串 (用于解码)
        encoder_.insert({line, i});
        decoder_[i] = line;
    }
    
    // 加载BPE合并规则，这些规则定义了令牌应该如何合并
    // 每个规则都是一个带有优先级排名的令牌对
    // 合并规则按优先级排序，优先级由行号决定(行号越小优先级越高)
    for (int i = 0; i < merge_len; i++) {
        std::getline(tok_file, line);
        // 找到合并规则中两个令牌之间的空格分隔符
        // 合并规则格式: "token1 token2"
        int d = line.find(" ");
        // 将合并规则存储为一对宽字符串及其优先级排名
        // 优先级由规则在文件中的行号决定
        bpe_ranks_.insert({{utf8_to_wstring(line.substr(0, d)),
            utf8_to_wstring(line.substr(d + 1))}, i});
    }
    
    // 创建字节到unicode和unicode到字节的映射表
    // 这是HuggingFace预处理的一部分，将所有可能的字节值映射到unicode字符
    // 以确保所有字节都能在分词过程中表示
    // 字节级预处理的必要性:
    // 1. 确保所有256个可能的字节值都能在分词过程中正确表示
    // 2. 避免特殊Unicode字符与实际文本字符冲突
    // 3. 统一处理所有语言和字符集
    
    // 将字节值范围插入unicode映射的函数
    // 用于将指定范围的字节值直接映射到相等的unicode值
    auto _insert_range = [=](int start, int end) {
        for (int c = start; c <= end; c++) {
            b2u_.insert({uint8_t(c), wchar_t(c)});
        }
    };

    b2u_.clear();
    // 将可打印ASCII字符直接映射到其unicode等价值
    // 这些字符在分词过程中保持原样，便于阅读和调试
    _insert_range(L'!', L'~');   // 基本ASCII可打印字符
    _insert_range(L'¡', L'¬');   // Latin-1补充字符
    _insert_range(L'®', L'ÿ');   // 更多Latin-1补充字符

    // 将剩余字节值(0-255)映射到从256开始的unicode字符
    // 对于没有直接映射的字节值，分配唯一的unicode字符
    int n = 0;
    for (int b = 0; b < 256; b++) {
        // 如果字节值尚未映射，则为其分配一个unicode字符
        // 分配从256开始的unicode字符，避免与可打印字符冲突
        if (b2u_.find(uint8_t(b)) == b2u_.end()) {
            b2u_.insert({uint8_t(b), wchar_t(256 + n)});
            n++;
        }
    }
    
    // 创建从unicode字符回到字节值的反向映射
    // 用于解码时将unicode字符转换回原始字节
    for (auto e : b2u_) {
        u2b_.insert({e.second, e.first});
    }
    return true;
}

// 从令牌生成所有相邻的字符对
// 这在BPE中用于识别潜在的合并操作
// BPE算法的第一步是识别所有相邻的字符对，这些对是可能的合并候选
// 例如，对于单词"low"，会生成字符对: ("l","o"), ("o","w")
// 这些字符对将用于后续的合并操作
void get_pairs(const std::wstring& word, std::vector<std::pair<std::wstring, std::wstring>>* pairs) {
    pairs->clear();

    // 需要至少2个字符才能形成一对
    // 如果单词长度小于2，则没有相邻字符对可以生成
    if (word.size() < 2) return;

    // 创建相邻字符对
    // 遍历单词中的每个字符(从第二个字符开始)，与前一个字符组成字符对
    wchar_t previous = word[0];
    for (int i = 1; i < word.size(); i++) {
        // 每对都由当前字符和前一个字符作为单字符字符串组成
        // 使用std::wstring(1, char)构造函数创建单字符的宽字符串
        pairs->push_back({std::wstring(1, previous), std::wstring(1, word[i])});
        previous = word[i];
    }
}

// 应用BPE(字节对编码)算法对词进行分词
// 这是HuggingFace BPE实现的核心，递归地合并字符对
// HuggingFace BPE算法特点:
// 1. 使用预定义的合并规则列表，每个规则有一个优先级排名
// 2. 在每一轮中，选择具有最高优先级(最低分数)的可合并对进行合并
// 3. 合并后更新相邻对信息，继续下一轮合并
// 4. 直到没有可合并的对为止
void HuggingfaceTokenizer::bpe(const std::wstring& token, const BPERanks& bpe_ranks, std::vector<std::wstring>* result) {
    // 集合用于跟踪哪些对已被合并以避免重新处理
    // merged集合记录已经合并的字符对的索引，避免重复处理
    std::set<int> merged;  // 记录已合并的对的索引
    
    // 查找未合并对的左邻居的辅助函数
    // 在合并操作后，需要找到某个位置的左邻居(即左边最近的未合并对)
    auto _left = [](int i, std::set<int>& merged) {
        for (int j = i - 1; j >= -1; j--) {
            if (merged.find(j) == merged.end()) return j;
        }
        return -1;
    };
    
    // 查找未合并对的右邻居的辅助函数
    // 在合并操作后，需要找到某个位置的右邻居(即右边最近的未合并对)
    auto _right = [](int i, int cap, std::set<int>& merged) {
        for (int j = i + 1; j < cap; j++) {
            if (merged.find(j) == merged.end()) return j;
        }
        return cap;
    };

    // 使用令牌中的所有相邻字符对初始化
    // 将输入令牌分解为相邻的字符对，作为BPE的起点
    std::vector<std::pair<std::wstring, std::wstring>> pairs;
    get_pairs(token, &pairs);

    // 主BPE合并循环 - 继续直到无法进行更多合并
    // 重复选择优先级最高的可合并对进行合并
    while (true) {
        int min_score = INT_MAX;  // 跟踪最高优先级合并(最低分数)
        int to_merge = -1;        // 要合并的对的索引

        // 查找具有最高优先级(最低分数)的对进行合并
        // 遍历所有字符对，找到优先级最高的可合并对
        for (int i = 0; i < pairs.size(); ++i) {
            // 跳过已合并的对
            if (merged.find(i) == merged.end()) {  // 对i未合并
                // 查找此对的合并优先级分数
                // bpe_ranks中存储了预定义的合并规则及其优先级
                auto iter = bpe_ranks.find(pairs[i]);
                int score = iter != bpe_ranks.end() ? iter->second : INT_MAX;
                // 跟踪最高优先级合并
                if (score < min_score) {
                    min_score = score;
                    to_merge = i;
                }
            }
        }

        // 如果无法进行更多合并，则跳出循环
        // 当找不到可合并的对时，说明BPE过程结束
        if (to_merge == -1) break;

        // 将选中的对标记为已合并
        merged.insert(to_merge);
        // 通过连接两个部分创建合并的令牌
        // 将选中的字符对合并为一个新的令牌
        std::wstring merge_into = pairs[to_merge].first + pairs[to_merge].second;

        // 更新相邻对以反映合并
        // 合并操作会影响相邻的字符对，需要更新它们的信息
        int l = _left(to_merge, merged);
        if (l >= 0) pairs[l].second = merge_into;  // 更新左邻居的右组件
        int r = _right(to_merge, pairs.size(), merged);
        if (r < pairs.size()) pairs[r].first = merge_into;  // 更新右邻居的左组件
    }  // 结束while (true)

    // 生成最终分词结果
    // 根据合并结果生成最终的分词序列
    if (merged.size() == pairs.size()) {
        // 如果所有对都已合并，则整个令牌是一个片段
        // 这种情况发生在令牌完全匹配某个预定义片段时
        result->push_back(token);

    } else {
        // 否则，收集未合并的对作为最终分词结果
        // 将所有未合并的字符对组合成最终的分词结果
        for (int i = 0; i < pairs.size(); ++i) {
            if (merged.find(i) == merged.end()) {
                // 对于序列中的第一个片段，包括左组件
                if (_left(i, merged) < 0) result->push_back(pairs[i].first);
                // 始终包括右组件
                result->push_back(pairs[i].second);
            }
        }
    }
}

// 使用HuggingFace BPE分词将字符串编码为令牌ID
// 这是一个多步骤过程，涉及正则表达式分割、字节编码、BPE和ID映射
// HuggingFace分词完整流程:
// 1. 使用正则表达式将输入文本分割成粗粒度的令牌
// 2. 将每个令牌的字节转换为预定义的unicode字符(字节级别预处理)
// 3. 对每个unicode令牌应用BPE算法进行细粒度分词
// 4. 将BPE分词结果转换回UTF-8字符串
// 5. 将最终的字符串令牌映射到对应的ID
void HuggingfaceTokenizer::encode(const std::string& str, std::vector<int>& ids) {
    // 用于将文本分割为令牌的正则表达式
    // 适当处理缩写、单词、数字、标点符号和空白
    // 正则表达式各部分含义:
    // 's|'t|'re|'ve|'m|'ll|'d  : 常见的英文缩写后缀
    //  ?[[:alpha:]]+             : 字母序列，可选前导空格
    //  ?[[:digit:]]+             : 数字序列，可选前导空格
    //  ?[^\\s\\w]+                 : 非字母数字字符，可选前导空格
    // \\s+                       : 连续空白字符
    std::regex re("('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\\s\\w]+|\\s+)");
    std::string input = str;
    std::vector<std::string> result;
    std::string token;
    std::smatch match;
    
    // 将每个正则表达式匹配项作为单独的令牌处理
    // 这是第一步粗粒度分割，将文本分解为较大的令牌单元
    while (std::regex_search(input, match, re)) {
        token = match.str(0);
        input = match.suffix().str();
        
        // 将令牌中的每个字节转换为其unicode表示
        // 这是HuggingFace的字节级别预处理步骤:
        // 1. 将令牌视为字节序列
        // 2. 使用预定义的映射表将每个字节映射到特定的unicode字符
        // 3. 这确保了所有可能的字节都能在后续处理中正确表示
        std::wstring wtoken;
        for (char c : token) {
            wtoken.push_back(b2u_.at(uint8_t(c)));
        }

        // 对unicode令牌应用BPE算法
        // 这是核心分词步骤，将粗粒度令牌进一步细分为子词单元
        std::vector<std::wstring> bpe_tokens;
        bpe(wtoken, bpe_ranks_, &bpe_tokens);

        // 将BPE令牌转换回UTF-8字符串
        // 将BPE处理后的unicode令牌转换回原始的UTF-8字符串表示
        for (auto ws : bpe_tokens) {
            result.push_back(wstring_to_utf8(ws));
        }
    }
    
    // 将最终令牌映射到其对应的ID
    // 通过编码器字典将字符串令牌转换为整数ID
    for (auto s : result) {
        ids.push_back(encoder_.at(s));
    }
}

// 将令牌ID解码回其原始字节序列
// 这逆转了分词期间执行的字节到unicode编码
std::string HuggingfaceTokenizer::decode(int id) {
    // printf("decode id = %d, %lu, %s#\n", id, decoder_.size(), decoder_.at(id).c_str());
    // 边界检查以防止数组访问错误
    if (id >= decoder_.size()) {
        return "";
    }
    
    // 获取此ID的令牌字符串
    std::wstring w = utf8_to_wstring(decoder_.at(id));
    std::string r;
    
    // 将每个unicode字符转换回其原始字节值
    for (wchar_t c : w) {
        // 查找此unicode字符的字节值
        if (u2b_.find(c) != u2b_.end()) {
            r.push_back(char(u2b_.at(c)));
        }
    }
    return r;
}
}
}
