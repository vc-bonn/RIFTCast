#include <riftcast/protocol/riftcastprotocol.h>

#include <Network/NetworkUtils.h>

namespace rift
{
namespace protocol
{
ProtocolVersion::ProtocolVersion() : major(RIFTCAST_PROTOCOL_VERSION_MAJOR), minor(RIFTCAST_PROTOCOL_VERSION_MINOR) {}

bool ProtocolVersion::isCompatible() const
{
    return RIFTCAST_PROTOCOL_VERSION_MAJOR == major;
}


ProtocolHeader::ProtocolHeader() : version(ProtocolVersion()), task(MessageTask::INVALID) {}

ProtocolHeader::ProtocolHeader(const MessageTask new_task) : version(ProtocolVersion()), task(new_task) {}

std::vector<uint8_t> createConnectionMessage()
{
    std::vector<uint8_t> response(sizeof(ProtocolHeader));
    auto header = ProtocolHeader(MessageTask::CONNECT_CLIENT);
    std::memcpy(response.data(), &header, sizeof(ProtocolHeader));

    return response;
}

std::vector<uint8_t> createDisconnectionMessage()
{
    std::vector<uint8_t> response(sizeof(ProtocolHeader));
    auto header = ProtocolHeader(MessageTask::DISCONNECT_CLIENT);
    std::memcpy(response.data(), &header, sizeof(ProtocolHeader));

    return response;
}

std::vector<uint8_t>
createRenderRequest(const uint32_t& width, const uint32_t& height, const glm::mat4& view, const glm::mat4& projection)
{
    constexpr uint32_t message_size = sizeof(ProtocolHeader) + 4 * sizeof(uint32_t) + 2 * sizeof(glm::mat4);

    std::vector<uint8_t> response(message_size);
    auto header = ProtocolHeader(MessageTask::REQUEST_IMAGE);
    std::memcpy(response.data(), &header, sizeof(ProtocolHeader));

    uint32_t offset = sizeof(ProtocolHeader);
    atcg::NetworkUtils::writeInt<uint32_t>(response.data(), offset, width);
    atcg::NetworkUtils::writeInt<uint32_t>(response.data(), offset, height);
    atcg::NetworkUtils::writeBuffer(response.data(), offset, (uint8_t*)glm::value_ptr(view), sizeof(glm::mat4));
    atcg::NetworkUtils::writeBuffer(response.data(), offset, (uint8_t*)glm::value_ptr(projection), sizeof(glm::mat4));

    return response;
}

std::vector<uint8_t> createStereoRenderRequest(const uint32_t& width,
                                               const uint32_t& height,
                                               const glm::mat4& view_left,
                                               const glm::mat4& projection_left,
                                               const glm::mat4& view_right,
                                               const glm::mat4& projection_right)
{
    constexpr uint32_t message_size = sizeof(ProtocolHeader) + 6 * sizeof(uint32_t) + 4 * sizeof(glm::mat4);

    std::vector<uint8_t> response(message_size);
    auto header = ProtocolHeader(MessageTask::REQUEST_STEREO);
    std::memcpy(response.data(), &header, sizeof(ProtocolHeader));

    uint32_t offset = sizeof(ProtocolHeader);
    atcg::NetworkUtils::writeInt<uint32_t>(response.data(), offset, width);
    atcg::NetworkUtils::writeInt<uint32_t>(response.data(), offset, height);
    atcg::NetworkUtils::writeBuffer(response.data(), offset, (uint8_t*)glm::value_ptr(view_left), sizeof(glm::mat4));
    atcg::NetworkUtils::writeBuffer(response.data(),
                                    offset,
                                    (uint8_t*)glm::value_ptr(projection_left),
                                    sizeof(glm::mat4));
    atcg::NetworkUtils::writeBuffer(response.data(), offset, (uint8_t*)glm::value_ptr(view_right), sizeof(glm::mat4));
    atcg::NetworkUtils::writeBuffer(response.data(),
                                    offset,
                                    (uint8_t*)glm::value_ptr(projection_right),
                                    sizeof(glm::mat4));

    return response;
}

std::vector<uint8_t> createNoUpdateMessage()
{
    std::vector<uint8_t> response(sizeof(ProtocolHeader));
    auto header = ProtocolHeader(MessageTask::NO_UPDATE);
    std::memcpy(response.data(), &header, sizeof(ProtocolHeader));

    return response;
}

std::vector<uint8_t> createUpdateMessage(const glm::mat4& inv_view_projection,
                                         const torch::Tensor& encoded_jpeg,
                                         const torch::Tensor& encoded_depth)
{
    uint32_t message_size = sizeof(ProtocolHeader) + 3 * sizeof(uint32_t) + sizeof(glm::mat4) + encoded_jpeg.numel() +
                            encoded_depth.numel();


    std::vector<uint8_t> response(message_size);
    auto header = ProtocolHeader(MessageTask::UPDATE);
    std::memcpy(response.data(), &header, sizeof(ProtocolHeader));

    uint32_t offset = sizeof(ProtocolHeader);
    atcg::NetworkUtils::writeBuffer(response.data(),
                                    offset,
                                    (uint8_t*)glm::value_ptr(inv_view_projection),
                                    sizeof(glm::mat4));
    atcg::NetworkUtils::writeBuffer(response.data(), offset, encoded_jpeg.data_ptr<uint8_t>(), encoded_jpeg.numel());
    atcg::NetworkUtils::writeBuffer(response.data(), offset, encoded_depth.data_ptr<uint8_t>(), encoded_depth.numel());

    return response;
}

std::vector<uint8_t> createUpdateMessage(const glm::mat4& inv_view_projection_left,
                                         const glm::mat4& inv_view_projection_right,
                                         const torch::Tensor& encoded_jpeg_left,
                                         const torch::Tensor& encoded_jpeg_right)
{
    uint32_t message_size = sizeof(ProtocolHeader) + 4 * sizeof(uint32_t) + sizeof(glm::mat4) +
                            encoded_jpeg_left.numel() + encoded_jpeg_right.numel();


    std::vector<uint8_t> response(message_size);
    auto header = ProtocolHeader(MessageTask::STEREO_UPDATE);
    std::memcpy(response.data(), &header, sizeof(ProtocolHeader));

    uint32_t offset = sizeof(ProtocolHeader);
    atcg::NetworkUtils::writeBuffer(response.data(),
                                    offset,
                                    (uint8_t*)glm::value_ptr(inv_view_projection_left),
                                    sizeof(glm::mat4));
    atcg::NetworkUtils::writeBuffer(response.data(),
                                    offset,
                                    (uint8_t*)glm::value_ptr(inv_view_projection_right),
                                    sizeof(glm::mat4));
    atcg::NetworkUtils::writeBuffer(response.data(),
                                    offset,
                                    encoded_jpeg_left.data_ptr<uint8_t>(),
                                    encoded_jpeg_left.numel());
    atcg::NetworkUtils::writeBuffer(response.data(),
                                    offset,
                                    encoded_jpeg_right.data_ptr<uint8_t>(),
                                    encoded_jpeg_right.numel());

    return response;
}


std::vector<uint8_t> createCameraRequest()
{
    std::vector<uint8_t> response(sizeof(ProtocolHeader));
    auto header = ProtocolHeader(MessageTask::CAMERA_DATA);
    std::memcpy(response.data(), &header, sizeof(ProtocolHeader));

    return response;
}

std::vector<uint8_t> createCameraResponse(const std::vector<glm::mat4>& views,
                                          const std::vector<glm::mat4>& projections)
{
    std::vector<uint8_t> response(sizeof(ProtocolHeader) + 2 * sizeof(uint32_t) + sizeof(glm::mat4) * views.size() +
                                  sizeof(glm::mat4) * projections.size());
    auto header = ProtocolHeader(MessageTask::CAMERA_DATA);
    std::memcpy(response.data(), &header, sizeof(ProtocolHeader));

    uint32_t offset = sizeof(ProtocolHeader);
    atcg::NetworkUtils::writeBuffer(response.data(), offset, (uint8_t*)views.data(), views.size() * sizeof(glm::mat4));
    atcg::NetworkUtils::writeBuffer(response.data(),
                                    offset,
                                    (uint8_t*)projections.data(),
                                    projections.size() * sizeof(glm::mat4));

    return response;
}

}    // namespace protocol
}    // namespace rift