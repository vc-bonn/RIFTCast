#pragma once

#include <Core/glm.h>
#include <Core/Memory.h>
#include <DataStructure/BufferView.h>
#include <DataStructure/Graph.h>
#include <Renderer/Texture.h>

#include <torch/types.h>

namespace rift
{
/**
 * @brief Compute vertex normals
 *
 * @param positions The vertex positions
 * @param vertex_normals The output buffer for the normals (Should be (nx3) where n is the number of vertices).
 * @param faces The face information
 */
void computeVertexNormals(const torch::Tensor& positions, torch::Tensor& vertex_normals, const torch::Tensor& faces);

/**
 * @brief This function computes a binary vector of visible primitives from the view that the texture was created.
 * This function should be used for the current view.
 *
 * @param texture The texture
 * @param width The width of the texture
 * @param height The height of the texture
 * @param num_primitives The number of primitives (mesh->n_faces())
 * @return A binary vector that encodes which primitives are visible.
 */
torch::Tensor computeVisiblePrimitives(const cudaTextureObject_t& texture,
                                       const uint32_t width,
                                       const uint32_t height,
                                       const uint32_t num_primitives);

/**
 * @brief This function computes a binary vector of visible primitives from each input view.
 * This function should be used for the VCI camera setup.
 *
 * @param textures The texture handles for the individual VCI cams
 * @param cam_valid_mask A binary vector that encodes which cameras should be used.
 * @param width The width of the texture
 * @param height The height of the texture
 * @param num_cameras The number of cameras.
 * @param num_primitives The number of primitives (mesh->n_faces())
 * @return A binary vector that encodes which primitives are visible.
 */
torch::Tensor computeVisiblePrimitivesBatched(const atcg::DeviceBuffer<cudaTextureObject_t>& textures,
                                              const torch::Tensor& cam_valid_mask,
                                              const uint32_t width,
                                              const uint32_t height,
                                              const uint32_t num_cameras,
                                              const uint32_t num_primitives);

/**
 * @brief Quantize the depth data for data transmission.
 * The quantization is given by
 * depth = atcg::Math::ndc2linearDepth(atcg::Math::uv2ndc(depth), n, f);
 * depth_quantized = (uint16_t)(depth * 5000.0f);
 *
 * @param depth_data The depth data
 * @param n The near plane
 * @param f The far plane
 * @return The quantized depth data
 */
torch::Tensor quantize(const torch::Tensor& depth_data, const float n, const float f);

/**
 * @brief This function unprojects a color image with given quantized depth information.
 * The resulting tensor is a [n, 15] float tensor that is compatible with atcg::Graph::vertices.
 *
 * @param rgb_image The rgb image
 * @param inv_view_projection The inverse view projection matrix
 * @param depth_map The quantized depth map
 * @return The Vertices
 */
torch::Tensor unprojectVerticesQuantized(const torch::Tensor& rgb_image,
                                         const glm::mat4& inv_view_projection,
                                         const torch::Tensor& depth_map);

/**
 * @brief This function unprojects a color image with given depth information.
 * The depth data is expected to be a OpenGL depth map with the same near and far plane as inv_view_projection.
 *
 * @param rgb_image The rgb image
 * @param inv_view_projection The inverse view projection matrix
 * @param depth_map The depth map
 * @return A tuple with the vertices positions and colors.
 */
std::tuple<torch::Tensor, torch::Tensor>
unprojectVertices(const torch::Tensor& rgb_image, const glm::mat4& inv_view_projection, const torch::Tensor& depth_map);

/**
 * @brief This function unprojects a color image with given depth information.
 * The depth data is expected to be a OpenGL depth map with the same near and far plane as inv_view_projection.
 *
 * @param rgb_image The rgb image
 * @param inv_view_projection The inverse view projection matrix
 * @param depth_map The depth map
 * @param normal_map The normal map (world space)
 * @return A tuple with the vertices positions, colors, and normals.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unprojectVertices(const torch::Tensor& rgb_image,
                                                                          const glm::mat4& inv_view_projection,
                                                                          const torch::Tensor& depth_map,
                                                                          const torch::Tensor& normal_map);
}    // namespace rift