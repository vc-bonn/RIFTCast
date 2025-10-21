#pragma once

#include <Core/Memory.h>
#include <DataStructure/TorchUtils.h>
#include <Renderer/Framebuffer.h>

#include <riftcast/DatasetImporter.h>

namespace rift
{

struct GeometryReconstruction
{
    torch::Tensor vertices;
    torch::Tensor faces;
    torch::Tensor normals;
    torch::Tensor visible_primitives;
    uint32_t current_frame;
};

/**
 * @brief This class handles geometry reconstruction.
 */
class GeometryModule
{
public:
    /**
     * @brief Default constructor
     */
    GeometryModule();

    /**
     * @brief Destructor
     */
    ~GeometryModule();

    /**
     * @brief Init the module.
     * This method should be called on the thread that uses this class as it initializes a renderer and context for the
     * current thread.
     *
     * @param device_idx The device index where the geometry computation should be used. This has to be 0 for
     * ATCG_HEADLESS=Off
     * @param dataloader The dataloader
     */
    void init(const uint32_t device_idx, const atcg::ref_ptr<rift::DatasetImporter>& dataloader);

    /**
     * @brief Do the geometry reconstruction
     *
     * @param model A model matrix that defines the capturing volume. model * (-1,-1,-1,1) should define the lower left
     * corner of the volume and its scale is the scale of the volume
     * @param cam_valid A binary vector that indicates which cameras should be used for the reconstruction
     * @param frame The frame index that should be reconstructed. If frame = -1 (default), the frame index will be
     * determined based on the data framerate.
     *
     * @return The reconstruction containing: Vertex positions, Vertex normals, faces, visible primitives (a (num_cams,
     * num_primitives) primary tensor that encodes which primitive is visible from which view), and the frame index).
     */
    GeometryReconstruction
    compute_geometry(const glm::mat4& model, const torch::Tensor& cam_valid, const uint32_t frame = -1);

    /**
     * @brief Get the primitive buffers for each camera
     *
     * @return The vector of primitive framebuffers
     */
    std::vector<atcg::ref_ptr<atcg::Framebuffer>> getPrimitiveBuffers() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace rift