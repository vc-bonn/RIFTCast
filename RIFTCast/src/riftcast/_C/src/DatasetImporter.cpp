#include <riftcast/DatasetImporter.h>

#include <DataStructure/TorchUtils.h>
#include <Network/NetworkUtils.h>

#include <iomanip>
#include <fstream>

using json = nlohmann::json;

namespace rift
{
DatasetHeader IO::readDatasetHeader(const nlohmann::json& data)
{
    DatasetHeader header;

    std::string type    = data["type"];
    std::string version = data.value("version", "1.0");
    header.version      = version;
    std::string major   = version.substr(0, version.find_first_of("."));

    if(major != "2")
    {
        ATCG_ERROR("Wrong version of dataset header. Got {0}, expected {1}!", version, "2.0");
        return header;
    }

    if(type == "VCI" || type == "VCI_REAL")
    {
        header.type = DatasetHeader::DatasetType::VCI;
    }
    else
    {
        ATCG_ERROR("Could not match {0} to a dataset type. Use either ATCG, VCI, or Panoptic as dataset type");
        return header;
    }

    header.name = data.value("name", "");
    if(data.contains("dataset"))
    {
        header.frame_count = data["dataset"].value("frame_count", 1);
        header.start_frame = data["dataset"].value("start_frame", 0);
        header.path        = data["dataset"]["path"];
        header.camera_path = data["dataset"]["camera_path"];
        if(data["dataset"].contains("to_world"))
        {
            std::vector<double> to_world_data = data["dataset"]["to_world"];
            header.to_world                   = glm::transpose(glm::make_mat4(to_world_data.data()));
        }
        header.flip_images = data["dataset"].value("flip_images", true);
        header.flip_masks  = data["dataset"].value("flip_masks", true);
    }

    if(data.contains("reconstructor"))
    {
        header.partial_masks      = data["reconstructor"].value("partial_masks", false);
        header.level              = data["reconstructor"].value("level", 10);
        header.reconstruction_gpu = data["reconstructor"].value("gpu", 1);
        header.enable_smoothing   = data["reconstructor"].value("smoothing", true);
        header.kernel_size        = data["reconstructor"].value("kernel_size", 9);
        header.sigma              = data["reconstructor"].value("sigma", 2.0f);
    }

    if(data.contains("server"))
    {
        header.server_ip   = data["server"].value("ip", "127.0.0.1");
        header.server_port = data["server"].value("port", 25565);
    }

    if(data.contains("volume"))
    {
        header.volume_scale    = data["volume"].value("scale", 1.0f);
        std::vector<float> pos = data["volume"]["position"];
        header.volume_position = glm::make_vec3(pos.data());
    }
    if(data.contains("capture_servers"))
    {
        header.trigger_ip         = data["capture_servers"].value("trigger", "");
        header.capture_server_ips = data["capture_servers"].value("ips", std::vector<std::string>());
    }

    if(data.contains("renderer"))
    {
        header.renderer_gpu = data["renderer"].value("gpu", 0);
    }

    if(data.contains("inpainting"))
    {
        header.enable_inpainting = data["inpainting"].value("enable", true);
        header.inpaint_path      = data["inpainting"].value("path", "");
    }

    return header;
}

DatasetHeader IO::readDatasetHeader(const std::string& path)
{
    json data;

    try
    {
        std::fstream f(path);
        data = json::parse(f);
    }
    catch(const std::exception& e)
    {
        ATCG_ERROR("Failed to load meta file '{0}'\n   {1}", path.c_str(), e.what());
        return DatasetHeader();
    }

    return readDatasetHeader(data);
}

std::vector<torch::Tensor> DatasetImporter::getMasks(const torch::Tensor& valid)
{
    static float frame_id = start_frame();
    frame_id += _timer.elapsedSeconds() * 30.0f;
    _timer.reset();
    int frame = ((int)frame_id) % num_frames();
    return getMasks(frame, valid);
}

VCIDatasetImporter::VCIDatasetImporter(const DatasetHeader& header) : DatasetImporter(header)
{
    importCameras(header.camera_path);
}

VCIDatasetImporter::~VCIDatasetImporter() {}


std::vector<torch::Tensor> VCIDatasetImporter::getMasks(const uint32_t frame_idx, const torch::Tensor& valid)
{
    std::vector<torch::Tensor> compressed(num_cameras());

    torch::Tensor host_valid = valid.to(torch::kCPU);
    int index                = 0;
    float total_size         = 0;
    for(int i = 0; i < num_cameras(); ++i)
    {
        if(host_valid.index({(int)i}).item<int>() == 0)
        {
            continue;
        }

        uint32_t id = _cameras[i].id;
        std::stringstream frame;
        frame << std::setfill('0') << std::setw(5) << frame_idx;
        std::string mask_path = _root_path + "/frame_" + frame.str() + "/mask/mask_" + std::to_string(id) + ".bin";
        std::ifstream fstream(mask_path, std::ios::in | std::ios::binary);

        // get its size:
        fstream.seekg(0, std::ios::end);
        std::streampos fileSize = fstream.tellg();
        fstream.seekg(0, std::ios::beg);

        total_size += (float)fileSize;

        // read the data:
        std::vector<uint8_t> fileData(fileSize);
        torch::Tensor data =
            torch::empty({(int)(fileData.size() / sizeof(int32_t))}, atcg::TensorOptions::int32HostOptions());
        fstream.read((char*)data.data_ptr(), fileSize);

        compressed[index++] = data.cuda();
        if(index >= num_cameras()) break;
    }

    _last_available_frame = frame_idx;

    _mask_logger.logSample(total_size);

    return compressed;
}

std::vector<std::vector<uint8_t>> VCIDatasetImporter::getImages(const uint32_t frame_idx, const torch::Tensor& valid)
{
    int32_t num_valid = torch::sum(valid).item<int32_t>();
    std::vector<std::vector<uint8_t>> file_data(num_valid);
    torch::Tensor host_valid = valid.to(torch::kCPU);
    int index                = 0;
    float total_size         = 0;
    for(uint32_t i = 0; i < num_cameras(); ++i)
    {
        if(host_valid.index({(int)i}).item<int>() == 0)
        {
            continue;
        }

        uint32_t id = _cameras[i].id;
        std::stringstream frame;
        frame << std::setfill('0') << std::setw(5) << frame_idx;
        std::string path = _root_path + "/frame_" + frame.str() + "/rgb/rgb_" + std::to_string(id) + ".jpeg";

        std::ifstream input(path, std::ios::in | std::ios::binary | std::ios::ate);

        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if(file_data[index].size() < file_size)
        {
            file_data[index].resize(file_size);
        }
        if(!input.read((char*)file_data[index].data(), file_size))
        {
            ATCG_ERROR("JPEGDecoder: Cannot read from file: {0}", path);
        }

        total_size += file_size;

        ++index;
        if(index >= num_valid) break;
    }

    _rgb_logger.logSample(total_size);

    //_last_available_frame = frame_idx;
    return file_data;
}

void VCIDatasetImporter::importCameras(const std::string& camera_config)
{
    json data;

    try
    {
        std::fstream f(_root_path + "/" + camera_config);
        data = json::parse(f);
    }
    catch(const std::exception& e)
    {
        ATCG_ERROR("Failed to load camera file '{0}'\n   {1}", (_root_path + "/" + camera_config).c_str(), e.what());
        return;
    }

    json camera_data = data["cameras"];
    std::vector<glm::mat4> host_view_projections;
    uint32_t i = 0;
    for(auto it = camera_data.begin(); it != camera_data.end(); ++it)
    {
        uint32_t camera_id = (*it)["camera_id"];
        std::string name   = (*it)["camera_model_name"];

        // if(name != "ximea_IMX530_RGB") continue;

        std::vector<double> extrinsics_data = (*it)["extrinsics"]["view_matrix"];
        std::vector<double> intrinsics_data = (*it)["intrinsics"]["camera_matrix"];

        std::vector<int> resolution = (*it)["intrinsics"]["resolution"];
        uint32_t width              = resolution[0];
        uint32_t height             = resolution[1];
        _width                      = width;
        _height                     = height;

        glm::mat4 extrinsic_matrix = glm::make_mat4(extrinsics_data.data());
        extrinsic_matrix           = glm::transpose(extrinsic_matrix);    // Because glm

        glm::mat4 cv_to_gl(1);
        cv_to_gl[0][0] = 1.0f;
        cv_to_gl[1][1] = -1.0f;
        cv_to_gl[2][2] = -1.0f;

        // gl_to_cv = cv_to_gl
        extrinsic_matrix = cv_to_gl * extrinsic_matrix * cv_to_gl * _header.to_world;

        float fx = (float)intrinsics_data[0];
        float fy = (float)intrinsics_data[4];
        float cx = (float)intrinsics_data[2];
        float cy = (float)intrinsics_data[5];
        float n  = 0.01f;
        float f  = 1000.0f;

        atcg::CameraIntrinsics intrinsics = atcg::CameraUtils::convert_from_opencv(fx, fy, cx, cy, n, f, width, height);

        atcg::CameraExtrinsics extrinsics(extrinsic_matrix);
        atcg::ref_ptr<atcg::PerspectiveCamera> cam = atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics);

        _cameras.push_back(CameraData {name + std::to_string(camera_id), width, height, camera_id, cam});

        host_view_projections.push_back(cam->getViewProjection());
    }

    _view_projection_tensor = atcg::createHostTensorFromPointer((float*)host_view_projections.data(),
                                                                {(int)host_view_projections.size(), 4, 4})
                                  .to(torch::kCUDA);

    _view_projection_tensor = _view_projection_tensor.transpose(1, 2);

    _num_cameras = host_view_projections.size();
}

atcg::ref_ptr<DatasetImporter> createDatasetImporter(const DatasetHeader& header)
{
    switch(header.type)
    {
        case DatasetHeader::DatasetType::VCI:
        {
            return atcg::make_ref<VCIDatasetImporter>(header);
        }
        break;
    }

    return nullptr;
}

}    // namespace rift