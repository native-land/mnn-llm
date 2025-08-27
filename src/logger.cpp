//
// logger.cpp
// 全局文件日志系统实现
//
// Created for MNN-LLM project
//

#include "logger.hpp"
#include <filesystem>
#include <cstring>

namespace MNN {
namespace Transformer {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

void Logger::init(const std::string& filename, LogLevel level) {
    std::lock_guard<std::mutex> lock(logMutex_);
    
    if (initialized_) {
        return; // 已经初始化过了
    }
    
    // 关闭之前可能打开的文件
    if (logFile_.is_open()) {
        logFile_.close();
    }
    
    // 打开日志文件
    logFile_.open(filename, std::ios::app);
    if (!logFile_.is_open()) {
        std::cerr << "Warning: Failed to open log file: " << filename << std::endl;
        return;
    }
    
    currentLevel_ = level;
    initialized_ = true;
    
    // 写入启动日志
    logFile_ << "\n=== MNN-LLM Log Started at " << getCurrentTime() << " ===" << std::endl;
    logFile_.flush();
}

void Logger::log(LogLevel level, const std::string& message, const char* file, int line) {
    if (!initialized_ || level < currentLevel_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(logMutex_);
    
    if (!logFile_.is_open()) {
        return;
    }
    
    // 格式: [时间] [级别] [文件:行号] 消息
    logFile_ << "[" << getCurrentTime() << "] "
             << "[" << getLevelString(level) << "] ";
    
    if (file && line > 0) {
        // 只显示文件名，不显示完整路径
        const char* filename = strrchr(file, '/');
        if (filename) {
            filename++; // 跳过 '/'
        } else {
            filename = file;
        }
        logFile_ << "[" << filename << ":" << line << "] ";
    }
    
    logFile_ << message << std::endl;
    
    // 错误级别的日志立即刷新
    if (level >= ERROR) {
        logFile_.flush();
    }
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(logMutex_);
    if (logFile_.is_open()) {
        logFile_.flush();
    }
}

void Logger::setLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(logMutex_);
    currentLevel_ = level;
}

std::string Logger::getCurrentTime() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

const char* Logger::getLevelString(LogLevel level) {
    switch (level) {
        case DEBUG: return "DEBUG";
        case INFO:  return "INFO ";
        case WARN:  return "WARN ";
        case ERROR: return "ERROR";
        default:    return "UNKNOWN";
    }
}

Logger::~Logger() {
    if (logFile_.is_open()) {
        logFile_ << "=== MNN-LLM Log Ended at " << getCurrentTime() << " ===" << std::endl;
        logFile_.close();
    }
}

} // namespace Transformer
} // namespace MNN