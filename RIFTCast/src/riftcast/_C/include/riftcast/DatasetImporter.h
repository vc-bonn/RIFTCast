#pragma once

#include <vector>
#include <torch/types.h>
#include <SFML/Network.hpp>
#include <Core/UUID.h>

#include <Core/Memory.h>
#include <DataStructure/Graph.h>
#include <DataStructure/Timer.h>
#include "CameraData.h"

#include <riftcast/BenchmarkLogger.h>

#include <json.hpp>

namespace rift
{
struct DatasetHeader
{
    enum class DatasetType
    {
        VCI = 4,    // Real captured VCI data (from disk)
    };

    DatasetType type;
    std::string version;
    std::string name;
    std::string path;
    std::string camera_path;
    std::string server_ip;
    uint32_t server_port;
    uint32_t frame_count      = 0;
    uint32_t start_frame      = 0;
    bool flip_images          = true;
    bool flip_masks           = true;
    glm::vec3 volume_position = glm::vec3(0);
    float volume_scale        = 1.0f;
    uint32_t level            = 9;
    bool partial_masks        = false;
    std::string trigger_ip;
    std::vector<std::string> capture_server_ips;
    glm::mat4 to_world          = glm::mat4(1);
    uint32_t renderer_gpu       = 0;
    uint32_t reconstruction_gpu = 1;
    bool enable_inpainting      = true;
    std::string inpaint_path    = "";
    bool enable_smoothing       = true;
    uint32_t kernel_size        = 9;
    float sigma                 = 2.0f;

    inline std::string type_to_string() const
    {
        switch(type)
        {
            case DatasetType::VCI:
                return "VCI";
        };
    }
};

namespace IO
{
/**
 * @brief Read the meta information about a dataset
 *
 * @param data The json file
 *
 * @return Meta information about the dataset
 */
DatasetHeader readDatasetHeader(const nlohmann::json& data);

/**
 * @brief Read the meta information about a dataset
 *
 * @param path The path to the metafile
 *
 * @return Meta information about the dataset
 */
DatasetHeader readDatasetHeader(const std::string& path);
}    // namespace IO

/**
 * @brief A class to load datasets
 */
class DatasetImporter
{
public:
    /**
     * @brief Create an instance of a dataset loader
     *
     * @param root_path The root path of the data
     * @param header The data header
     */
    DatasetImporter(const DatasetHeader& header) : _root_path(header.path), _header(header)
    {
        auto t  = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream base;

        base << "bin/log/" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");

        auto uuid = atcg::UUID();

        _rgb_logger  = rift::BenchmarkLogger("rgb_" + std::to_string(uuid), base.str() + "_rgb.txt");
        _mask_logger = rift::BenchmarkLogger("mask_" + std::to_string(uuid), base.str() + "_mask.txt");
    }

    /**
     * @brief Destructor
     */
    virtual ~DatasetImporter() {}

    /**
     * @brief Get the masks
     * This function has a counter that gives masks with the capture speed (live demo) or 30fps for prerecorded data.
     *
     * @param valid A binary array that indicates which cameras should be fetched
     * @return A vector of compressed masks that can be decompressed with the maskcompression module
     */
    virtual std::vector<torch::Tensor> getMasks(const torch::Tensor& valid);

    /**
     * @brief Get the masks
     *
     * @param frame_idx The frame index
     * @param valid A binary array that indicates which cameras should be fetched
     * @return A vector of compressed masks that can be decompressed with the maskcompression module
     */
    virtual std::vector<torch::Tensor> getMasks(const uint32_t frame_idx, const torch::Tensor& valid) = 0;

    /**
     * @brief Get color images compressed as jpeg representation.
     *
     * @param frame_idx The frame index
     * @param valid A binary array that indicates which cameras should be fetched
     * @return A vector of vectors representing jpeg encoded images.
     */
    virtual std::vector<std::vector<uint8_t>> getImages(const uint32_t frame_idx, const torch::Tensor& valid) = 0;

    /**
     * @brief Get the last available frame index.
     * This value gets updated each time getMasks(valid) is called.
     *
     * @return The frame index of the last mask fetch
     */
    inline uint32_t getLastAvailableFrame() const { return _last_available_frame; }

    /**
     * @brief Get a (n, 4, 4) tensor with camera projection matrices
     *
     * @return The tensor
     */
    inline torch::Tensor getViewProjectionTensor() const { return _view_projection_tensor; }

    /**
     * @brief Get the camera data
     *
     * @return The cameras
     */
    inline std::vector<CameraData> getCameras() const { return _cameras; }

    /**
     * @brief Get the number of cameras
     *
     * @return The number of cameras
     */
    inline uint32_t num_cameras() const { return _num_cameras; }

    /**
     * @brief Get the width of the images.
     * All images should have the same resolution
     * @return The width
     */
    inline uint32_t width() const { return _width; }

    /**
     * @brief Get the height of the images.
     * All images should have the same resolution
     * @return The height
     */
    inline uint32_t height() const { return _height; }

    /**
     * @brief Get the number of frames in the sequence
     *
     * @return The number of frames
     */
    inline uint32_t num_frames() const { return _header.frame_count; }

    /**
     * @brief Get the start frame
     *
     * @return The start frame
     */
    inline uint32_t start_frame() const { return _header.start_frame; }

    /**
     * @brief If images should be flipped
     *
     * @return True if rgb images should be flipped
     */
    inline uint32_t flip_images() const { return _header.flip_images; }

    /**
     * @brief If masks should be flipped
     *
     * @return True if masks should be flipped
     */
    inline uint32_t flip_masks() const { return _header.flip_masks; }

    /**
     * @brief Level that should be used for the reconstruction
     *
     * @return The level
     */
    inline uint32_t level() const { return _header.level; }

    /**
     * @brief If partial masks should be used
     *
     * @return If partial masks should be used
     */
    inline bool partial_masks() const { return _header.partial_masks; }

    /**
     * @brief Get the gpu index of the gpu that should do the rendering
     *
     * @return The gpu index
     */
    inline uint32_t renderer_gpu() const { return _header.renderer_gpu; }

    /**
     * @brief Get the gpu index of the gpu that should do the reconstruction
     *
     * @return The gpu index
     */
    inline uint32_t reconstruction_gpu() const { return _header.reconstruction_gpu; }

    /**
     * @brief Inpainting enabled
     *
     * @return True if inpainting should be performed
     */
    inline bool enable_inpainting() const { return _header.enable_inpainting; }

    /**
     * @brief The path to the inpainting module
     *
     * @return The path
     */
    inline const std::string& inpainting_path() const { return _header.inpaint_path; }

    /**
     * @brief If mask smoothing should be enabled
     *
     * @return True if enabled
     */
    inline bool enable_smoothing() const { return _header.enable_smoothing; }

    /**
     * @brief The kernel size used for smoothing
     *
     * @return The kernel size
     */
    inline uint32_t kernel_size() const { return _header.kernel_size; }

    /**
     * @brief The sigma used for smoothing
     *
     * @return sigma
     */
    inline float sigma() const { return _header.sigma; }

protected:
    /**
     * @brief Import a set of cameras
     *
     * @param camera_config The filename of the camera relative to the root directory of the importer
     */
    virtual void importCameras(const std::string& camera_config) = 0;

    std::string _root_path;
    DatasetHeader _header;

    torch::Tensor _view_projection_tensor;
    std::vector<CameraData> _cameras;

    uint32_t _num_cameras;
    uint32_t _width;
    uint32_t _height;

    uint32_t _last_available_frame = -1;

    atcg::Timer _timer;

    rift::BenchmarkLogger _rgb_logger;
    rift::BenchmarkLogger _mask_logger;
};

class VCIDatasetImporter : public DatasetImporter
{
public:
    VCIDatasetImporter(const DatasetHeader& header);

    virtual ~VCIDatasetImporter();

    virtual std::vector<torch::Tensor> getMasks(const uint32_t frame_idx, const torch::Tensor& valid) override;

    virtual std::vector<std::vector<uint8_t>> getImages(const uint32_t frame_idx, const torch::Tensor& valid) override;

protected:
    virtual void importCameras(const std::string& camera_config) override;
};

atcg::ref_ptr<DatasetImporter> createDatasetImporter(const DatasetHeader& header);

}    // namespace rift