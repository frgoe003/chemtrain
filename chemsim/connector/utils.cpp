//
// Created by Paul Fuchs on 26.10.24.
//

#include "utils.h"

#include <functional>
#include <string>
#include <iostream>

namespace jcn {

    Logger::Logger() {
        // Reads the loglevel once

        char* loglevel_env = std::getenv("JCN_LOGLEVEL");
        if (loglevel_env == nullptr) return;

        loglevel = std::stoi(loglevel_env);

    }

    // Logs a message with a given log level
    void Logger::log(LogLevel level, const std::string& message)
    {

        // Loglevel not big enough
        if (!log(level)) return;

        std::cout << "[" << levelToString(level) << "] " << message << std::endl;

    }

    // Logs a message with a given log level
    bool Logger::log(LogLevel level)
    {

        return level <= static_cast<int>(loglevel);

    }

    std::string Logger::levelToString(LogLevel level) {
        switch (level) {
            case DEBUG:
                return "DEBUG";
            case INFO:
                return "INFO";
            case WARNING:
                return "WARNING";
            case ERROR:
                return "ERROR";
            case CRITICAL:
                return "CRITICAL";
            default:
                return "DEBUG";
        }
    }


} // namespace jcn
