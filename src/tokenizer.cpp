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
    while (in_len-- && ( str[in_] != '=') && is_base64(str[in_])) {
        char_array_4[i++] = str[in_]; in_++;
        if (i ==4) {
            // 将base64字符转换为6位值
            for (i = 0; i <4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }
            // 将4组6位重新组合成3个字节
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
    if (i) {
        // 用零填充
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
// 格式: 每行包含base64编码的令牌、分数和片段类型
bool Sentencepiece::load_vocab(std::ifstream& tok_file) {
    std::string line, token;
    // 从第一行读取词汇表大小
    std::getline(tok_file, line);
    int vocab_len = std::stoi(line);
    float score;
    int type;
    // 调整句子片段向量大小以容纳所有词汇项目
    sentence_pieces_.resize(vocab_len);
    // 加载每个词汇项目
    for (int index = 0; index < vocab_len; index++) {
        std::getline(tok_file, line);
        std::istringstream line_str(line);
        // 从行中解析令牌、分数和类型
        line_str >> token >> score >> type;
        // 将base64编码的令牌解码回原始字符串
        token = base64_decode(token);
        // 将整数类型转换为PieceType枚举
        auto piece_type = static_cast<PieceType>(type);
        // 创建带有令牌、分数和类型的SentencePiece对象
        SentencePiece piece = {token, score, piece_type};
        sentence_pieces_[index] = std::move(piece);
        // 根据类型对片段进行分类
        if (piece_type == PieceType::NORMAL) {
            // 正常片段存储在pieces_映射中以便快速查找
            pieces_.insert({token, index});
        } else {
            // 特殊片段(控制、未知等)存储在reserved_id_map_中
            reserved_id_map_.insert({token, index});
            if (piece_type == PieceType::UNKNOWN) {
                // 跟踪未知令牌ID
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
// 参考: https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc
Sentencepiece::EncodeResult Sentencepiece::bpe_encode(string_view_ normalized, float alpha) {
    // 表示具有合并分数的相邻符号对的结构
    struct SymbolPair {
        int left;     // 此对的左索引
        int right;    // 此对的右索引
        float score;  // 此对的分数。分数越大越好。
        size_t size;  // 此片段的长度
    };

    // 优先队列中符号对优先级的比较器
    // 分数越高优先级越高，左索引用作决胜局
    class SymbolPairComparator {
    public:
        const bool operator()(SymbolPair *h1, SymbolPair *h2) {
            return (h1->score < h2->score || (h1->score == h2->score && h1->left > h2->left));
        }
    };

    // 表示分割过程中符号的结构
    struct Symbol {
        int prev;     // 此符号的前一个索引。BOS为-1。
        int next;     // 此符号的下一个索引。EOS为-1。
        bool freeze = false;  // 此符号永远不会被合并。
        string_view_ piece;   // 此符号的实际文本内容
    };

    // 按合并分数管理符号对的优先队列
    using Agenda = std::priority_queue<SymbolPair *, std::vector<SymbolPair *>, SymbolPairComparator>;
    Agenda agenda;
    // 在分割过程中保存所有符号的向量
    std::vector<Symbol> symbols;
    symbols.reserve(normalized.size());
    // 用于未使用片段重新分割的反向合并规则
    // 将合并的符号映射到其组成部分
    std::unordered_map<string_view_, std::pair<string_view_, string_view_>> rev_merge;
    // 符号对对象的存储以避免重复分配
    std::vector<std::unique_ptr<SymbolPair>> symbol_pair_holder;
    
    // 检查合并两个相邻符号是否会创建有效片段的函数
    // 如果是，则将该对添加到议程中并附带其分数
    auto MaybeAddNewSymbolPair = [this, &symbol_pair_holder, &symbols, &agenda, &rev_merge](int left, int right) {
        // 如果任一符号无效或已冻结则跳过
        if (left == -1 || right == -1 || symbols[left].freeze || symbols[right].freeze) {
            return;
        }
        // 从两个符号创建合并的片段
        const string_view_ piece(symbols[left].piece.data(), symbols[left].piece.size() + symbols[right].piece.size());
        std::string piece_str(piece.to_string());
        // 检查合并的片段是否存在于我们的词汇表中
        const auto it = pieces_.find(piece_str);
        if (it == pieces_.end()) {
            return;
        }
        // 创建新的符号对并将其添加到议程中
        symbol_pair_holder.emplace_back(new SymbolPair);
        auto *h = symbol_pair_holder.back().get();
        h->left = left;
        h->right = right;
        h->score = get_score(it->second);  // 获取此片段的分数
        h->size = piece.size();
        agenda.push(h);

        // 存储未使用片段重新分割的反向合并信息
        if (is_unused(it->second)) {
            rev_merge[piece] = std::make_pair(symbols[left].piece, symbols[right].piece);
        }
    };
    
    // 将输入分割成初始字符级符号
    int index = 0;
    while (!normalized.empty()) {
        Symbol s;
        // 确定当前UTF-8字符的长度
        int mblen = std::min<int>(normalized.size(), one_char_len(normalized.data()));
        s.piece = string_view_(normalized.data(), mblen);
        // 为相邻符号设置链表指针
        s.prev = index == 0 ? -1 : index - 1;
        normalized.remove_prefix(mblen);
        s.next = normalized.empty() ? -1 : index + 1;
        ++index;
        symbols.emplace_back(s);
    }

    // 如果未创建符号则返回空结果
    if (symbols.empty()) {
        return {};
    }
    
    // 使用所有相邻符号对初始化议程
    for (size_t i = 1; i < symbols.size(); ++i) {
        MaybeAddNewSymbolPair(i - 1, i);
    }

    // BPE-dropout正则化机制 (https://arxiv.org/pdf/1910.13267.pdf)
    std::mt19937 rand_gen;
    auto skip_merge = [&]() {
        // 以alpha概率跳过合并(0.0 = 不dropout, 1.0 = 总是dropout)
        if (alpha <= 0.0) return false;
        if (alpha >= 1.0) return true;
        std::uniform_real_distribution<> gen(0.0, 1.0);
        return gen(rand_gen) < alpha;
    };

    // 主BPE合并循环
    while (!agenda.empty()) {
        // 从议程中获取分数最高的对
        SymbolPair *top = agenda.top();
        agenda.pop();

        // 如果此对不再有效(符号已被合并)则跳过
        if (symbols[top->left].piece.empty() || symbols[top->right].piece.empty() ||
            symbols[top->left].piece.size() + symbols[top->right].piece.size() != top->size) {
            continue;
        }

        // 如果启用则应用BPE-dropout
        if (skip_merge()) continue;
        
        // 将两个符号合并为一个
        symbols[top->left].piece = string_view_(
                                                symbols[top->left].piece.data(),
                                                symbols[top->left].piece.size() + symbols[top->right].piece.size());

        // 更新链表指针以移除右侧符号
        symbols[top->left].next = symbols[top->right].next;
        if (symbols[top->right].next >= 0) {
            symbols[symbols[top->right].next].prev = top->left;
        }
        symbols[top->right].piece = string_view_("");

        // 添加由此合并操作创建的新潜在合并
        MaybeAddNewSymbolPair(symbols[top->left].prev, top->left);
        MaybeAddNewSymbolPair(top->left, symbols[top->left].next);
    }

    // 将未使用片段递归重新分割成更小的已知片段的函数
    std::function<void(string_view_, EncodeResult*)> resegment;
    resegment = [this, &resegment, &rev_merge](string_view_ w, EncodeResult *output) -> void {
        std::string w_str(w.to_string());
        const int id = piece_to_id(w_str);
        // 如果片段有效且未使用，则将其添加到输出中
        if (id == -1 || !is_unused(id)) {
            output->emplace_back(w, id);
            return;
        }
        // 对于未使用片段，查找如何分解它们
        const auto p = rev_merge.find(w);
        if (p == rev_merge.end()) {
            // 此块永远不会被调用，因为`rev_merge`存储了所有未使用ID的
            // 重新分割信息。
            output->emplace_back(w, id);
            return;
        }
        // 递归重新分割组成部分
        resegment(p->second.first, output);
        resegment(p->second.second, output);
    };
    
    // 通过遍历符号链表生成最终编码结果
    EncodeResult output;
    for (int index = 0; index != -1; index = symbols[index].next) {
        resegment(symbols[index].piece, &output);
    }
    return output;
}

// 使用SentencePiece算法将字符串编码为令牌ID
// 如果启用则使用字节级回退处理未知令牌
void Sentencepiece::encode(const std::string& str, std::vector<int>& ids) {
    // 执行BPE编码以获取片段-字符串对
    auto result = bpe_encode(str);
    size_t consumed = 0;
    // 将每个片段转换为其对应的ID
    for (const auto &p : result) {
        const string_view_ w = p.first;   // 片段
        const int id = p.second;          // ID
        const bool is_unk = (id == unk_id_);
        // 如果启用且存在未知令牌则使用字节级回退处理
        if (is_unk && byte_fall_back_) {
            // 将未知片段分解为UTF-8字节
            for (int i = 0; i < w.size(); ++i) {
                // 创建字节片段表示
                const char b = w[i];
                const auto piece = byte_to_piece(b);
                auto sp_id = piece_to_id(piece);
                ids.push_back(sp_id);
            }
        } else {
            // 添加正常令牌ID
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
void Tiktoken::encode(const std::string& str, std::vector<int>& ids) {
    // 空字符串的早期返回
    if (str.empty()) {
        return;
    }
    size_t i = 0;
    // 从左到右处理字符串
    while (i < str.size()) {
        bool found_pair = false;
        // 尝试在当前位置匹配最长可能的符号
        size_t longest_match_len = 0;
        std::string longest_match;

        // 检查递减长度的子字符串以找到最长匹配
        // 这确保我们总是使用最长可能的令牌
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
// 连续片段使用"##"前缀标识
std::vector<int> BertTokenizer::word_piece(const std::string& token) {
    // 首先检查整个令牌是否存在于词汇表中
    auto it = encoder_.find(token);
    if (it != encoder_.end()) {
        return {it->second};
    }
    
    // 如果未找到，则分解为子词片段
    std::vector<int> ids;
    std::string current = token;
    while (!current.empty()) {
        int match_id = -1;      // 匹配片段的ID
        size_t match_pos = 0;   // 匹配结束的位置
        
        // 尝试在当前字符串开头匹配最长可能的片段
        for (int len = current.size(); len > 0; --len) {
            std::string candidate = current.substr(0, len);
            // 为连续片段添加"##"前缀(除了第一个片段)
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
// 这涉及两个步骤:
// 1. 预分词: 将文本分割为基本令牌(单词、标点符号等)
// 2. WordPiece分词: 根据需要进一步细分令牌
void BertTokenizer::encode(const std::string& str, std::vector<int>& ids) {
    // 第一遍: 将文本预分词为基本令牌
    std::vector<std::string> tokens;
    std::string current_token;
    size_t i = 0;
    
    // 处理每个字符以构建基本令牌
    while (i < str.size()) {
        current_token.clear();
        unsigned char c = static_cast<unsigned char>(str[i]);
        
        // 处理多字节UTF-8字符(简化处理3字节字符)
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
        else if (isalnum(c)) {
            // 将连续的字母数字字符分组并转换为小写
            while (i < str.size() && isalnum(static_cast<unsigned char>(str[i]))) {
                current_token += tolower(str[i]);
                ++i;
            }
        }
        // 处理标点符号和符号
        else if (ispunct(c)) {
            current_token = str[i];
            ++i;
        }
        // 处理空格、制表符、回车
        else if (isspace(c)) {
            ++i;
            continue;  // 跳过空白字符
        }
        // 处理任何其他单字节字符
        else {
            current_token = str[i];
            ++i;
        }
        
        // 将非空令牌添加到列表中
        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }
    }

    // 第二遍: 对每个基本令牌应用WordPiece分词
    for (auto token : tokens) {
        // 根据需要将每个令牌细分为子词片段
        for (auto id : word_piece(token)) {
            ids.push_back(id);
        }
    }
}

// 将UTF-8编码的字符串转换为宽字符串(wstring)
// 使用标准库codecvt功能进行UTF-8到UTF-32/UTF-16转换
std::wstring utf8_to_wstring(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.from_bytes(str);
}

// 将宽字符串(wstring)转换为UTF-8编码的字符串
// 使用标准库codecvt功能进行UTF-32/UTF-16到UTF-8转换
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
// 格式:
// - 第一行: 词汇表大小和合并规则计数
// - 接下来的vocab_len行: 词汇表令牌
// - 接下来的merge_len行: 合并规则(令牌对)
bool HuggingfaceTokenizer::load_vocab(std::ifstream& tok_file) {
    std::string line, token;
    // 从第一行读取词汇表和合并规则计数
    int vocab_len, merge_len;
    std::getline(tok_file, line);
    std::istringstream line_str(line);
    line_str >> vocab_len >> merge_len;
    
    // 将词汇表令牌加载到编码器/解码器映射中
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
    for (int i = 0; i < merge_len; i++) {
        std::getline(tok_file, line);
        // 找到合并规则中两个令牌之间的空格分隔符
        int d = line.find(" ");
        // 将合并规则存储为一对宽字符串及其优先级排名
        bpe_ranks_.insert({{utf8_to_wstring(line.substr(0, d)),
            utf8_to_wstring(line.substr(d + 1))}, i});
    }
    
    // 创建字节到unicode和unicode到字节的映射表
    // 这是HuggingFace预处理的一部分，将所有可能的字节值映射到unicode字符
    // 以确保所有字节都能在分词过程中表示
    
    // 将字节值范围插入unicode映射的函数
    auto _insert_range = [=](int start, int end) {
        for (int c = start; c <= end; c++) {
            b2u_.insert({uint8_t(c), wchar_t(c)});
        }
    };

    b2u_.clear();
    // 将可打印ASCII字符直接映射到其unicode等价值
    _insert_range(L'!', L'~');   // 基本ASCII可打印字符
    _insert_range(L'¡', L'¬');   // Latin-1补充字符
    _insert_range(L'®', L'ÿ');   // 更多Latin-1补充字符

    // 将剩余字节值(0-255)映射到从256开始的unicode字符
    int n = 0;
    for (int b = 0; b < 256; b++) {
        // 如果字节值尚未映射，则为其分配一个unicode字符
        if (b2u_.find(uint8_t(b)) == b2u_.end()) {
            b2u_.insert({uint8_t(b), wchar_t(256 + n)});
            n++;
        }
    }
    
    // 创建从unicode字符回到字节值的反向映射
    for (auto e : b2u_) {
        u2b_.insert({e.second, e.first});
    }
    return true;
}

// 从令牌生成所有相邻的字符对
// 这在BPE中用于识别潜在的合并操作
void get_pairs(const std::wstring& word, std::vector<std::pair<std::wstring, std::wstring>>* pairs) {
    pairs->clear();

    // 需要至少2个字符才能形成一对
    if (word.size() < 2) return;

    // 创建相邻字符对
    wchar_t previous = word[0];
    for (int i = 1; i < word.size(); i++) {
        // 每对都由当前字符和前一个字符作为单字符字符串组成
        pairs->push_back({std::wstring(1, previous), std::wstring(1, word[i])});
        previous = word[i];
    }
}

// 应用BPE(字节对编码)算法对词进行分词
// 这是HuggingFace BPE实现的核心，递归地合并字符对
void HuggingfaceTokenizer::bpe(const std::wstring& token, const BPERanks& bpe_ranks, std::vector<std::wstring>* result) {
    // 集合用于跟踪哪些对已被合并以避免重新处理
    std::set<int> merged;  // 记录已合并的对的索引
    
    // 查找未合并对的左邻居的辅助函数
    auto _left = [](int i, std::set<int>& merged) {
        for (int j = i - 1; j >= -1; j--) {
            if (merged.find(j) == merged.end()) return j;
        }
        return -1;
    };
    
    // 查找未合并对的右邻居的辅助函数
    auto _right = [](int i, int cap, std::set<int>& merged) {
        for (int j = i + 1; j < cap; j++) {
            if (merged.find(j) == merged.end()) return j;
        }
        return cap;
    };

    // 使用令牌中的所有相邻字符对初始化
    std::vector<std::pair<std::wstring, std::wstring>> pairs;
    get_pairs(token, &pairs);

    // 主BPE合并循环 - 继续直到无法进行更多合并
    while (true) {
        int min_score = INT_MAX;  // 跟踪最高优先级合并(最低分数)
        int to_merge = -1;        // 要合并的对的索引

        // 查找具有最高优先级(最低分数)的对进行合并
        for (int i = 0; i < pairs.size(); ++i) {
            // 跳过已合并的对
            if (merged.find(i) == merged.end()) {  // 对i未合并
                // 查找此对的合并优先级分数
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
        if (to_merge == -1) break;

        // 将选中的对标记为已合并
        merged.insert(to_merge);
        // 通过连接两个部分创建合并的令牌
        std::wstring merge_into = pairs[to_merge].first + pairs[to_merge].second;

        // 更新相邻对以反映合并
        int l = _left(to_merge, merged);
        if (l >= 0) pairs[l].second = merge_into;  // 更新左邻居的右组件
        int r = _right(to_merge, pairs.size(), merged);
        if (r < pairs.size()) pairs[r].first = merge_into;  // 更新右邻居的左组件
    }  // 结束while (true)

    // 生成最终分词结果
    if (merged.size() == pairs.size()) {
        // 如果所有对都已合并，则整个令牌是一个片段
        result->push_back(token);

    } else {
        // 否则，收集未合并的对作为最终分词结果
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
void HuggingfaceTokenizer::encode(const std::string& str, std::vector<int>& ids) {
    // 用于将文本分割为令牌的正则表达式
    // 适当处理缩写、单词、数字、标点符号和空白
    std::regex re("('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\\s\\w]+|\\s+)");
    std::string input = str;
    std::vector<std::string> result;
    std::string token;
    std::smatch match;
    
    // 将每个正则表达式匹配项作为单独的令牌处理
    while (std::regex_search(input, match, re)) {
        token = match.str(0);
        input = match.suffix().str();
        
        // 将令牌中的每个字节转换为其unicode表示
        std::wstring wtoken;
        for (char c : token) {
            wtoken.push_back(b2u_.at(uint8_t(c)));
        }

        // 对unicode令牌应用BPE算法
        std::vector<std::wstring> bpe_tokens;
        bpe(wtoken, bpe_ranks_, &bpe_tokens);

        // 将BPE令牌转换回UTF-8字符串
        for (auto ws : bpe_tokens) {
            result.push_back(wstring_to_utf8(ws));
        }
    }
    
    // 将最终令牌映射到其对应的ID
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
