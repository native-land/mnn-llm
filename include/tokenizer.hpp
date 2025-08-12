//
//  tokenizer.hpp
//
//  Created by MNN on 2023/09/25.
//  ZhaodeWang
//

#ifndef TOKENIZER_hpp
#define TOKENIZER_hpp

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <iostream>
// #include <string_view>
#include <cstring>
class string_view_ {
public:
    string_view_() : data_(nullptr), size_(0) {}
    string_view_(const char* data) : data_(data), size_(std::strlen(data)) {}
    string_view_(const char* data, std::size_t size) : data_(data), size_(size) {}
    string_view_(const std::string& str) : data_(str.data()), size_(str.size()) {}
    constexpr string_view_(const string_view_&) noexcept = default;
    string_view_& operator=(const string_view_&) noexcept = default;
    const char& operator[](size_t pos) const { return data_[pos]; }
    constexpr const char* data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr bool empty() const { return size_ == 0; }
    std::string to_string() const { return std::string(data_, size_); }
    bool operator==(const string_view_& other) const noexcept {
        return size_ == other.size_ && strncmp(data_, other.data_, size_) == 0;
    }
    void remove_prefix(size_t n) {
        if (n < size_) {
            data_ += n;
            size_ -= n;
        } else {
            data_ = "";
            size_ = 0;
        }
    }
private:
    const char* data_;
    std::size_t size_ = 0;
};
// std::string_view impl in c++11 end

namespace std {
    template<>
    class hash<string_view_> {
    public:
        size_t operator()(const string_view_& sv) const {
            size_t result = 0;
            for (size_t i = 0; i < sv.size(); ++i) {
                result = (result * 31) + static_cast<size_t>(sv[i]);
            }
            return result;
        }
    };
}
namespace MNN {
namespace Transformer {
// std::string_view impl in c++11 start

// 分词器基类
class Tokenizer {
public:
    // 魔数，用于标识文件格式
    static constexpr int MAGIC_NUMBER = 430;
    
    // 分词器类型枚举
    enum TokenizerType {
        SENTENCEPIECE = 0,  // SentencePiece分词器
        TIKTOIKEN = 1,      // Tiktoken分词器
        BERT = 2,           // BERT分词器
        HUGGINGFACE = 3     // HuggingFace分词器
    };
    
    // 默认构造函数
    Tokenizer() = default;
    
    // 虚析构函数
    virtual ~Tokenizer() = default;
    
    // 根据文件名创建分词器实例
    static Tokenizer* createTokenizer(const std::string& filename);
    
    // 判断是否为停止标记
    bool is_stop(int token);
    
    // 判断是否为特殊标记
    bool is_special(int token);
    
    // 对字符串进行编码（分词）
    std::vector<int> encode(const std::string& str);
    
    // 解码标记ID为字符串（纯虚函数，子类必须实现）
    virtual std::string decode(int id) = 0;
    
protected:
    // 加载特殊标记
    virtual void load_special(std::ifstream& file);
    
    // 加载词汇表（纯虚函数，子类必须实现）
    virtual bool load_vocab(std::ifstream& file) = 0;
    
    // 执行编码操作（纯虚函数，子类必须实现）
    virtual void encode(const std::string& str, std::vector<int>& ids) = 0;
    
    // 特殊标记列表
    std::vector<int> special_tokens_;
    
    // 停止标记列表
    std::vector<int> stop_tokens_;
    
    // 前缀标记列表
    std::vector<int> prefix_tokens_;
};

// SentencePiece分词器实现
class Sentencepiece : public Tokenizer {
public:
    // 默认构造函数
    Sentencepiece() = default;
    
    // 解码标记ID为字符串
    virtual std::string decode(int id) override;
    
protected:
    // 加载词汇表
    virtual bool load_vocab(std::ifstream& file) override;
    
    // 执行编码操作
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
    
private:
    // 模型类型枚举
    enum ModelType {
        UNIGRAM = 1,  // Unigram模型
        BPE = 2,      // BPE模型
        WORD = 3,     // 词模型
        CHAR = 4      // 字符模型
    };
    
    // 分词片段类型枚举
    enum PieceType {
        NORMAL = 1,       // 普通片段
        UNKNOWN = 2,      // 未知片段
        CONTROL = 3,      // 控制片段
        USER_DEFINED = 4, // 用户定义片段
        UNUSED = 5,       // 未使用片段
        BYTE = 6          // 字节片段
    };
    
    // 分词片段结构体
    struct SentencePiece {
        std::string piece;     // 片段字符串
        float score;           // 分数
        PieceType type = PieceType::NORMAL;  // 片段类型
        
        // 默认构造函数
        SentencePiece() {}
        
        // 带参数构造函数
        SentencePiece(const std::string& p, float s, PieceType t) : piece(p), score(s), type(t) {}
    };
    
    // 编码结果类型定义
    using EncodeResult = std::vector<std::pair<string_view_, int>>;
    
private:
    // 模型训练类型
    ModelType type_ = BPE;
    
    // 是否启用字节回退
    bool byte_fall_back_ = true;
    
    // 未知标记ID
    int unk_id_ = 0;
    
    // 模型中的所有分词片段
    std::vector<SentencePiece> sentence_pieces_;
    
    // 普通片段的映射表（片段->ID）
    std::unordered_map<std::string, int> pieces_;
    
    // 控制、未知和字节片段的映射表（片段->ID）
    std::unordered_map<std::string, int> reserved_id_map_;
    
private:
    // 获取指定ID片段的分数
    float get_score(int id) const;
    
    // 判断指定ID片段是否未使用
    bool is_unused(int id) const;
    
    // 判断指定ID片段是否为控制片段
    bool is_control(int id) const;
    
    // 将片段转换为ID
    int piece_to_id(const std::string& w) const;
    
    // 将字节转换为片段
    std::string byte_to_piece(unsigned char c) const;
    
    // BPE编码实现
    EncodeResult bpe_encode(string_view_ str, float alpha = 0.f);
};

// Tiktoken分词器实现
class Tiktoken : public Tokenizer {
public:
    // 默认构造函数
    Tiktoken() = default;
    
    // 解码标记ID为字符串
    virtual std::string decode(int id) override;
    
protected:
    // 加载词汇表
    virtual bool load_vocab(std::ifstream& file) override;
    
    // 执行编码操作
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
    
    // 编码器映射表（字符串->ID）
    std::unordered_map<std::string, int> encoder_;
    
    // 解码器映射表（ID->字符串）
    std::vector<std::string> decoder_;
};

// BERT分词器实现
class BertTokenizer : public Tiktoken {
public:
    // 默认构造函数
    BertTokenizer() = default;
    
protected:
    // 执行编码操作
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
    
private:
    // 词片分割函数
    std::vector<int> word_piece(const std::string& token);
};

// HuggingFace分词器实现
class HuggingfaceTokenizer : public Tokenizer {
    // 宽字符串对的哈希函数
    struct hash_pair_wstring {
        size_t operator()(const std::pair<std::wstring, std::wstring>& p) const {
            auto hash1 = std::hash<std::wstring>{}(p.first);
            auto hash2 = std::hash<std::wstring>{}(p.second);
            // 如果hash1等于hash2，则它们的异或为零
            return (hash1 != hash2) ? hash1 ^ hash2 : hash1;
        }
    };
    
    // BPE等级映射表类型定义
    using BPERanks = std::unordered_map<std::pair<std::wstring, std::wstring>, int, hash_pair_wstring>;
    
public:
    // 默认构造函数
    HuggingfaceTokenizer() = default;
    
    // 解码标记ID为字符串
    virtual std::string decode(int id) override;
    
protected:
    // 加载词汇表
    virtual bool load_vocab(std::ifstream& file) override;
    
    // 执行编码操作
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
    
private:
    // BPE算法实现
    void bpe(const std::wstring& token, const BPERanks& bpe_ranks, std::vector<std::wstring>* result);
    
    // BPE等级映射表
    BPERanks bpe_ranks_;
    
    // 字节到宽字符映射表
    std::unordered_map<uint8_t, wchar_t> b2u_;
    
    // 宽字符到字节映射表
    std::unordered_map<wchar_t, uint8_t> u2b_;
    
    // 编码器映射表（字符串->ID）
    std::unordered_map<std::string, int> encoder_;
    
    // 解码器映射表（ID->字符串）
    std::vector<std::string> decoder_;
};
};
};

#endif // TOKENIZER_hpp
