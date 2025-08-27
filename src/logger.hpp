//
// logger.hpp
// 全局文件日志系统
//
// Created for MNN-LLM project
//

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <memory>
#include <string>

namespace MNN {
namespace Transformer {

/**
 * @brief 日志级别枚举
 */
enum LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3
};

/**
 * @brief 全局文件日志类
 * 
 * 该类提供线程安全的文件日志功能，支持不同日志级别，
 * 可以在整个程序中使用而不影响标准输出。
 */
class Logger {
public:
    /**
     * @brief 获取日志单例实例
     */
    static Logger& getInstance();
    
    /**
     * @brief 初始化日志系统
     * 
     * @param filename 日志文件名，默认为"mnn_llm.log"
     * @param level 日志级别，默认为INFO
     */
    void init(const std::string& filename = "mnn_llm.log", LogLevel level = INFO);
    
    /**
     * @brief 写入日志
     * 
     * @param level 日志级别
     * @param message 日志消息
     * @param file 源文件名（可选）
     * @param line 源文件行号（可选）
     */
    void log(LogLevel level, const std::string& message, 
             const char* file = nullptr, int line = -1);
    
    /**
     * @brief 刷新日志缓冲区
     */
    void flush();
    
    /**
     * @brief 设置日志级别
     */
    void setLevel(LogLevel level);
    
    /**
     * @brief 获取当前时间字符串
     */
    std::string getCurrentTime();

private:
    Logger() = default;
    ~Logger();
    
    // 禁用拷贝构造和赋值
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    std::ofstream logFile_;      ///< 日志文件流
    LogLevel currentLevel_;      ///< 当前日志级别
    std::mutex logMutex_;        ///< 线程安全互斥锁
    bool initialized_;           ///< 是否已初始化
    
    /**
     * @brief 获取日志级别字符串
     */
    const char* getLevelString(LogLevel level);
};

} // namespace Transformer
} // namespace MNN

// 便捷的宏定义，自动包含文件名和行号
#define LOG_DEBUG(msg) MNN::Transformer::Logger::getInstance().log(MNN::Transformer::DEBUG, msg, __FILE__, __LINE__)
#define LOG_INFO(msg) MNN::Transformer::Logger::getInstance().log(MNN::Transformer::INFO, msg, __FILE__, __LINE__)
#define LOG_WARN(msg) MNN::Transformer::Logger::getInstance().log(MNN::Transformer::WARN, msg, __FILE__, __LINE__)
#define LOG_ERROR(msg) MNN::Transformer::Logger::getInstance().log(MNN::Transformer::ERROR, msg, __FILE__, __LINE__)

// 带格式化的日志宏
#define LOG_DEBUG_F(fmt, ...) do { \
    std::ostringstream oss; \
    oss << fmt; \
    MNN::Transformer::Logger::getInstance().log(MNN::Transformer::DEBUG, oss.str(), __FILE__, __LINE__); \
} while(0)

#define LOG_INFO_F(fmt, ...) do { \
    std::ostringstream oss; \
    oss << fmt; \
    MNN::Transformer::Logger::getInstance().log(MNN::Transformer::INFO, oss.str(), __FILE__, __LINE__); \
} while(0)

#define LOG_WARN_F(fmt, ...) do { \
    std::ostringstream oss; \
    oss << fmt; \
    MNN::Transformer::Logger::getInstance().log(MNN::Transformer::WARN, oss.str(), __FILE__, __LINE__); \
} while(0)

#define LOG_ERROR_F(fmt, ...) do { \
    std::ostringstream oss; \
    oss << fmt; \
    MNN::Transformer::Logger::getInstance().log(MNN::Transformer::ERROR, oss.str(), __FILE__, __LINE__); \
} while(0)

#endif // LOGGER_HPP