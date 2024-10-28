//
// Created by Paul Fuchs on 26.10.24.
//

#ifndef UTILS_H
#define UTILS_H

#include <functional>
#include <string>
#include <iostream>


namespace jcn {

    // Enum to represent log levels
    enum LogLevel {
        DEBUG = 2,
        INFO = 1,
        WARNING = 0,
        ERROR = -1,
        CRITICAL = -2
    };

    class Logger {
    public:
        ~Logger() = default;

        // Singleton instance
        static Logger& getlogger() {
            static Logger instance;
            return instance;
        }

        // Logs a message with a given log level
        void log(LogLevel level, const std::string& message);

        // Checks if the log level is sufficiently high
        bool log(LogLevel level);

    private:
        Logger();

        int loglevel = 1;

        std::string levelToString(LogLevel level);

    };

} // namespace jcn



#endif //UTILS_H
