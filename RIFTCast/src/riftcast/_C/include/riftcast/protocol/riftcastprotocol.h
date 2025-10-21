#pragma once

#include <json.hpp>
#include <Core/glm.h>
#include <torch/types.h>

namespace rift
{
namespace protocol
{
/**
 * @brief Version of the protocol.
 * @note Increase RIFTCAST_PROTOCOL_VERSION_MAJOR if any of the payloads change (except by adding new ones).
 * Increase RIFTCAST_PROTOCOL_VERSION_MINOR if any new payloads or message types are added. Set to 0 if
 * RIFTCAST_PROTOCOL_VERSION_MAJOR increases.
 */
#define RIFTCAST_PROTOCOL_VERSION_MAJOR 1
#define RIFTCAST_PROTOCOL_VERSION_MINOR 3

enum class MessageTask : int32_t
{
    CONNECT_CLIENT    = 1,    // A client connects to the server
    DISCONNECT_CLIENT = 2,    // A client disconnects from the server
    REQUEST_IMAGE     = 3,    // Request an image from the current view position
    NO_UPDATE         = 4,    // The server did not update the image
    UPDATE            = 5,    // A new image is rendered and was sent to the client
    MASKS             = 6,
    MASKS_IMAGES      = 7,
    REQUEST_STEREO    = 8,
    STEREO_UPDATE     = 9,
    CAMERA_DATA       = 10,
    INVALID           = -1    // Invalid task
};

struct ProtocolVersion
{
    ProtocolVersion();

    bool isCompatible() const;

    uint16_t major;
    uint16_t minor;
};

struct ProtocolHeader
{
    ProtocolHeader();

    explicit ProtocolHeader(const MessageTask new_task);

    ProtocolVersion version;
    MessageTask task;
};

/**
 * @brief Create connect message
 *
 * @return The resulting message
 */
std::vector<uint8_t> createConnectionMessage();

/**
 * @brief Create disconnect message
 *
 * @return The resulting message
 */
std::vector<uint8_t> createDisconnectionMessage();

/**
 * @brief Create render request message
 *
 * @param width The image width
 * @param height The image height
 * @param view The view matrix
 * @param projection The projection matrix
 *
 * @return The resulting message
 */
std::vector<uint8_t>
createRenderRequest(const uint32_t& width, const uint32_t& height, const glm::mat4& view, const glm::mat4& projection);

/**
 * @brief Create stereo render request message
 *
 * @param width The image width
 * @param height The image height
 * @param view_left The left view matrix
 * @param projection_left The left projection matrix
 * @param view_right The left view matrix
 * @param projection_right The left projection matrix
 *
 * @return The resulting message
 */
std::vector<uint8_t> createStereoRenderRequest(const uint32_t& width,
                                               const uint32_t& height,
                                               const glm::mat4& view_left,
                                               const glm::mat4& projection_left,
                                               const glm::mat4& view_right,
                                               const glm::mat4& projection_right);

/**
 * @brief Create no update message
 *
 * @return The resulting message
 */
std::vector<uint8_t> createNoUpdateMessage();

/**
 * @brief Create an update message
 *
 * @param inv_view_projection The inverse view projection matrix
 * @param encoded_jpeg The encoded jpeg image
 * @param encoded_depth The encoded depth
 *
 * @return The resulting message
 */
std::vector<uint8_t> createUpdateMessage(const glm::mat4& inv_view_projection,
                                         const torch::Tensor& encoded_jpeg,
                                         const torch::Tensor& encoded_depth);

/**
 * @brief Create an update message
 *
 * @param inv_view_projection_left The left inverse view projection matrix
 * @param inv_view_projection_right The right inverse view projection matrix
 * @param encoded_jpeg_left The left encoded jpeg image
 * @param encoded_jpeg_right The right encoded encoded jpeg image
 *
 * @return The resulting message
 */
std::vector<uint8_t> createUpdateMessage(const glm::mat4& inv_view_projection_left,
                                         const glm::mat4& inv_view_projection_right,
                                         const torch::Tensor& encoded_jpeg_left,
                                         const torch::Tensor& encoded_jpeg_right);

std::vector<uint8_t> createCameraRequest();

std::vector<uint8_t> createCameraResponse(const std::vector<glm::mat4>& views,
                                          const std::vector<glm::mat4>& projections);

}    // namespace protocol
}    // namespace rift