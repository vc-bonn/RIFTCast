#pragma once

#include <Core/Memory.h>
#include <torch/types.h>

namespace rift
{
/**
 * @brief A class to handle inpainting
 */
class InpaintingModule
{
public:
    /**
     * @brief Default constructor
     */
    InpaintingModule();

    /**
     * @brief Destructor
     */
    ~InpaintingModule();

    /**
     * @brief Constructor
     *
     * @param base_path Base path to the compiled DSTT model
     */
    InpaintingModule(const std::string& base_path);

    /**
     * @brief Inpaint a frame. It is assumed that this frame is part of a continouos video sequence
     *
     * @param image The input image
     * @param mask The mask with to be inpainted regions
     *
     * @return The inpainted frame
     */
    torch::Tensor inpaint(const torch::Tensor& image, const torch::Tensor& mask);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace rift