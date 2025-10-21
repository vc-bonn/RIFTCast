#pragma once

#include <Core/Memory.h>
#include <riftcast/DatasetImporter.h>
#include <DataStructure/TorchUtils.h>
#include <Renderer/Framebuffer.h>
#include <Renderer/ShaderManager.h>
#include <DataStructure/JPEGDecoder.h>

#include <riftcast/GeometryModule.h>

namespace rift
{
/**
 * @brief This class handles rendering.
 */
class RenderModule
{
public:
    /**
     * @brief Default constructor
     */
    RenderModule();

    /**
     * @brief Destructor
     */
    ~RenderModule();

    /**
     * @brief Init the module.
     * This method should be called on the thread that uses this class as it initializes a renderer and context for the
     * current thread.
     *
     * @param device_idx The device index where the rendering should be done. This has to be 0 for
     * ATCG_HEADLESS=Off
     * @param dataloader The dataloader
     * @param backend The JPEG decoding backend
     */
    void init(const uint32_t device_idx,
              const atcg::ref_ptr<rift::DatasetImporter>& dataloader,
              const atcg::JPEGBackend& backend);

    /**
     * @brief Update the state of the renderer with new geometry.
     *
     * @param reconstruction The reconstruction
     * @param camera The camera that is used to select textures
     * @param width The width of the output frame
     * @param height The height of the output frame
     */
    void updateState(const GeometryReconstruction& reconstruction,
                     const atcg::ref_ptr<atcg::PerspectiveCamera>& camera,
                     const uint32_t width,
                     const uint32_t height);

    /**
     * @brief Render a frame
     *
     * @param camera The camera from which to render the scene (this does not have to be the same as for updateState()
     * but deviating too much may lead to rendering artifacts).
     * @return The output framebuffer with the following color attachements:
     * 0 - RGBA Color information
     * 1 - INT Entity IDs (for the framework, not used by the reconstruction)
     * 2 - INT Visibility information (-1 - background; i in [0,1,2,3] how many cameras where used to reconstruct the
     * texture) 3 - FLOAT depth buffer in OpenGL Space in [0,1]
     */
    atcg::ref_ptr<atcg::Framebuffer> renderFrame(const atcg::ref_ptr<atcg::PerspectiveCamera>& camera);

    /**
     * @brief Get the binary tensor of cameras that have been chosen for texture reconstruction
     *
     * @return The tensor
     */
    torch::Tensor getChosenCameraIndices() const;

    /**
     * @brief Get the shader manager object
     *
     * @return The shader manager
     */
    atcg::ref_ptr<atcg::ShaderManagerSystem> getShaderManager() const;

    /**
     * @brief Set the background color
     *
     * @param color The background color
     */
    void setBackgroundColor(const glm::vec4& color);

    void setGreedySelection(const bool use_greedy_selection);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace rift