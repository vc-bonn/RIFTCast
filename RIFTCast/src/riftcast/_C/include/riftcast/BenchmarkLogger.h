#pragma once

#include <Core/Log.h>
#include <Core/Memory.h>

namespace rift
{
class BenchmarkLogger
{
public:
    BenchmarkLogger() = default;

    BenchmarkLogger(const std::string& name, const std::string& output_path);

    void logSample(const float sample);

private:
    atcg::ref_ptr<spdlog::logger> _logger;
};
}    // namespace rift