#pragma once

#include <Renderer/PerspectiveCamera.h>

namespace rift
{

/**
 * @brief A struct to hold camera information
 */
struct CameraData
{
    std::string name;
    uint32_t width, height;
    uint32_t id;
    atcg::ref_ptr<atcg::PerspectiveCamera> cam;
};
}    // namespace rift