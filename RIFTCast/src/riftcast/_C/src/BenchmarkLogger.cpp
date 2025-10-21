#include <riftcast/BenchmarkLogger.h>

#include "spdlog/sinks/basic_file_sink.h"

namespace rift
{
BenchmarkLogger::BenchmarkLogger(const std::string& name, const std::string& output_path)
{
    _logger = spdlog::basic_logger_mt(name, output_path);
    _logger->set_level(spdlog::level::trace);
    _logger->set_pattern("%H-%M-%S.%e, %v");
}

void BenchmarkLogger::logSample(const float sample)
{
    _logger->trace(sample);
}
}    // namespace rift