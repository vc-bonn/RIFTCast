#include <torch/python.h>

#include <json.hpp>
#include <pybind11_json.hpp>
#include <pybind11/stl.h>

#include <riftcast/DatasetImporter.h>

#include <riftcast/protocol/vciprotocol.h>

using namespace pybind11::literals;

namespace detail
{
static nlohmann::json getCameras(const std::shared_ptr<rift::DatasetImporter>& importer)
{
    auto cameras = importer->getCameras();

    nlohmann::json data;

    for(auto camera: cameras)
    {
        nlohmann::json camera_json;

        camera_json["name"]        = camera.name;
        camera_json["width"]       = camera.width;
        camera_json["height"]      = camera.height;
        auto P                     = camera.cam->getProjection();
        auto V                     = camera.cam->getView();
        camera_json["cam"]["P"]    = std::vector<float>(glm::value_ptr(P), glm::value_ptr(P) + 16);
        camera_json["cam"]["V"]    = std::vector<float>(glm::value_ptr(V), glm::value_ptr(V) + 16);
        camera_json["cam"]["near"] = camera.cam->getNear();
        camera_json["cam"]["far"]  = camera.cam->getFar();

        data[std::to_string(camera.id)] = camera_json;
    }

    return data;
}
}    // namespace detail

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    auto m_datasetheader = py::class_<rift::DatasetHeader>(m, "DatasetHeader");
    auto m_datasettype   = py::enum_<rift::DatasetHeader::DatasetType>(m, "DatasetType");
    auto m_datasetimporter =
        py::class_<rift::DatasetImporter, std::shared_ptr<rift::DatasetImporter>>(m, "DatasetImporter");
    auto m_atcg_importer =
        py::class_<rift::ATCGDatasetImporter, rift::DatasetImporter, std::shared_ptr<rift::ATCGDatasetImporter>>(m,
                                                                                                                 "ATCGD"
                                                                                                                 "ata"
                                                                                                                 "setI"
                                                                                                                 "mport"
                                                                                                                 "e"
                                                                                                                 "r");
    auto m_vci_importer =
        py::class_<rift::VCIDatasetImporter, rift::DatasetImporter, std::shared_ptr<rift::VCIDatasetImporter>>(m,
                                                                                                               "VCIData"
                                                                                                               "set"
                                                                                                               "Imp"
                                                                                                               "orter");
    auto m_vcireal_importer =
        py::class_<rift::VCIRealDatasetImporter, rift::DatasetImporter, std::shared_ptr<rift::VCIRealDatasetImporter>>(
            m,
            "VCIRealDa"
            "tasetImpo"
            "rter");
    auto m_vcirealtcp_importer = py::class_<rift::VCIRealTCPDatasetImporter,
                                            rift::DatasetImporter,
                                            std::shared_ptr<rift::VCIRealTCPDatasetImporter>>(m,
                                                                                              "VCIRea"
                                                                                              "lTCPDa"
                                                                                              "tasetI"
                                                                                              "mporte"
                                                                                              "r");
    auto m_panoptic_importer   = py::class_<rift::PanopticDatasetImporter,
                                            rift::DatasetImporter,
                                            std::shared_ptr<rift::PanopticDatasetImporter>>(m,
                                                                                          "Panoptic"
                                                                                            "DatasetI"
                                                                                            "mporte"
                                                                                            "r");
    auto m_tcp_importer =
        py::class_<rift::TCPDatasetImporter, rift::DatasetImporter, std::shared_ptr<rift::TCPDatasetImporter>>(m,
                                                                                                               "TCPData"
                                                                                                               "set"
                                                                                                               "Imp"
                                                                                                               "orter");
    auto m_io = m.def_submodule("IO");

    auto m_protocol = m.def_submodule("protocol");

    auto m_message_task = py::enum_<rift::protocol::MessageTask>(m_protocol, "MessageTask");

    auto m_protocolversion = py::class_<rift::protocol::ProtocolVersion>(m_protocol, "ProtocolVersion");

    auto m_protocolheader = py::class_<rift::protocol::ProtocolHeader>(m_protocol, "ProtocolHeader");

    m_datasetheader.def(py::init<>())
        .def_readwrite("type", &rift::DatasetHeader::type)
        .def_readwrite("version", &rift::DatasetHeader::version)
        .def_readwrite("name", &rift::DatasetHeader::name)
        .def_readwrite("path", &rift::DatasetHeader::path)
        .def_readwrite("camera_path", &rift::DatasetHeader::camera_path)
        .def_readwrite("server_ip", &rift::DatasetHeader::server_ip)
        .def_readwrite("server_port", &rift::DatasetHeader::server_port)
        .def_readwrite("frame_count", &rift::DatasetHeader::frame_count)
        .def_readwrite("start_frame", &rift::DatasetHeader::start_frame)
        .def_readwrite("floor_offset", &rift::DatasetHeader::floor_offset)
        .def_readwrite("flip_images", &rift::DatasetHeader::flip_images)
        .def_readwrite("flip_masks", &rift::DatasetHeader::flip_masks)
        .def_readwrite("level", &rift::DatasetHeader::level)
        .def_readwrite("partial_masks", &rift::DatasetHeader::partial_masks)
        .def_readwrite("apply_transform", &rift::DatasetHeader::apply_transform)
        .def_property(
            "volume_position",
            [](const rift::DatasetHeader& header)
            {
                std::array<float, 3> arr = {header.volume_position.x,
                                            header.volume_position.y,
                                            header.volume_position.z};
                return arr;
            },
            [](rift::DatasetHeader& header, const std::array<float, 3>& arr)
            { header.volume_position = glm::vec3(arr[0], arr[1], arr[2]); })
        .def_readwrite("volume_scale", &rift::DatasetHeader::volume_scale)
        .def("json",
             [](const rift::DatasetHeader& header)
             {
                 nlohmann::json json_file;

                 json_file["type"]                   = header.type_to_string();
                 json_file["version"]                = header.version;
                 json_file["dataset"]["frame_count"] = header.frame_count;
                 json_file["dataset"]["start_frame"] = header.start_frame;
                 json_file["dataset"]["path"]        = header.path;
                 json_file["dataset"]["camera_path"] = header.camera_path;
                 json_file["dataset"]["flip_images"] = header.flip_images;
                 json_file["dataset"]["flip_masks"]  = header.flip_masks;

                 json_file["reconstructor"]["floor_offset"]  = header.floor_offset;
                 json_file["reconstructor"]["partial_masks"] = header.partial_masks;
                 json_file["reconstructor"]["level"]         = header.level;

                 json_file["volume"]["position"] = {header.volume_position.x,
                                                    header.volume_position.y,
                                                    header.volume_position.z};
                 json_file["volume"]["scale"]    = header.volume_scale;

                 json_file["server"]["ip"]   = header.server_ip;
                 json_file["server"]["port"] = header.server_port;

                 return json_file;
             })
        .def_static("create_empty",
                    []()
                    {
                        nlohmann::json json_file;

                        json_file["type"]                   = "None";
                        json_file["version"]                = "2.2";
                        json_file["dataset"]["frame_count"] = 0;
                        json_file["dataset"]["start_frame"] = 0;
                        json_file["dataset"]["path"]        = "";
                        json_file["dataset"]["camera_path"] = "";
                        json_file["dataset"]["flip_images"] = true;
                        json_file["dataset"]["flip_masks"]  = true;

                        json_file["reconstructor"]["floor_offset"]  = 0;
                        json_file["reconstructor"]["level"]         = 10;
                        json_file["reconstructor"]["partial_masks"] = false;

                        json_file["volume"]["position"] = {0, 0, 0};
                        json_file["volume"]["scale"]    = 1;

                        json_file["server"]["ip"]   = "127.0.0.1";
                        json_file["server"]["port"] = 25565;

                        return json_file;
                    });

    m_datasettype.value("ATCG", rift::DatasetHeader::DatasetType::ATCG)
        .value("VCI_REAL_TCP", rift::DatasetHeader::DatasetType::VCI_REAL_TCP)
        .value("VCI", rift::DatasetHeader::DatasetType::VCI)
        .value("PANOPTIC", rift::DatasetHeader::DatasetType::PANOPTIC)
        .value("TCP", rift::DatasetHeader::DatasetType::TCP)
        .value("VCI_REAL", rift::DatasetHeader::DatasetType::VCI_REAL)
        .export_values();

    m_io.def("readDatasetHeader",
             [](const py::dict& dict)
             {
                 nlohmann::json data = dict;
                 return rift::IO::readDatasetHeader(data);
             })
        .def("readDatasetHeader", [](const std::string& string) { return rift::IO::readDatasetHeader(string); });

    m_io.def("createDatasetImporter", rift::createDatasetImporter);

    m_datasetimporter
        .def("getMasks",
             (std::vector<torch::Tensor> (rift::DatasetImporter::*)(
                 const torch::Tensor&))&rift::DatasetImporter::getMasks)
        .def("getMasks",
             (std::vector<torch::Tensor> (
                 rift::DatasetImporter::*)(const uint32_t, const torch::Tensor&))&rift::DatasetImporter::getMasks)
        .def("getImages", &rift::DatasetImporter::getImages)
        .def("getViewProjectionTensor", &rift::DatasetImporter::getViewProjectionTensor)
        .def("num_cameras", &rift::DatasetImporter::num_cameras)
        .def("width", &rift::DatasetImporter::width)
        .def("height", &rift::DatasetImporter::height)
        .def("num_frames", &rift::DatasetImporter::num_frames)
        .def("floor_offset", &rift::DatasetImporter::floor_offset)
        .def("start_frame", &rift::DatasetImporter::start_frame)
        .def("flip_images", &rift::DatasetImporter::flip_images)
        .def("flip_masks", &rift::DatasetImporter::flip_masks)
        .def("partial_masks", &rift::DatasetImporter::partial_masks)
        .def("level", &rift::DatasetImporter::level)
        .def("getCameras",
             [](const std::shared_ptr<rift::DatasetImporter>& importer)
             {
                 pybind11::dict data = detail::getCameras(importer);
                 return data;
             });

    m_atcg_importer.def(py::init<const rift::DatasetHeader&>())
        .def("getMasks",
             (std::vector<torch::Tensor> (rift::ATCGDatasetImporter::*)(
                 const torch::Tensor&))&rift::ATCGDatasetImporter::getMasks)
        .def("getMasks",
             (std::vector<torch::Tensor> (rift::ATCGDatasetImporter::*)(
                 const uint32_t,
                 const torch::Tensor&))&rift::ATCGDatasetImporter::getMasks)
        .def("getImages", &rift::ATCGDatasetImporter::getImages)
        .def("getViewProjectionTensor", &rift::ATCGDatasetImporter::getViewProjectionTensor)
        .def("num_cameras", &rift::ATCGDatasetImporter::num_cameras)
        .def("width", &rift::ATCGDatasetImporter::width)
        .def("height", &rift::ATCGDatasetImporter::height)
        .def("num_frames", &rift::ATCGDatasetImporter::num_frames)
        .def("floor_offset", &rift::ATCGDatasetImporter::floor_offset)
        .def("start_frame", &rift::ATCGDatasetImporter::start_frame)
        .def("flip_images", &rift::ATCGDatasetImporter::flip_images)
        .def("flip_masks", &rift::ATCGDatasetImporter::flip_masks)
        .def("partial_masks", &rift::ATCGDatasetImporter::partial_masks)
        .def("level", &rift::ATCGDatasetImporter::level)
        .def("getCameras",
             [](const std::shared_ptr<rift::ATCGDatasetImporter>& importer)
             {
                 pybind11::dict data = detail::getCameras(importer);
                 return data;
             });

    m_vci_importer.def(py::init<const rift::DatasetHeader&>())
        .def("getMasks",
             (std::vector<torch::Tensor> (rift::VCIDatasetImporter::*)(
                 const torch::Tensor&))&rift::VCIDatasetImporter::getMasks)
        .def("getMasks",
             (std::vector<torch::Tensor> (
                 rift::VCIDatasetImporter::*)(const uint32_t, const torch::Tensor&))&rift::VCIDatasetImporter::getMasks)
        .def("getImages", &rift::VCIDatasetImporter::getImages)
        .def("getViewProjectionTensor", &rift::VCIDatasetImporter::getViewProjectionTensor)
        .def("num_cameras", &rift::VCIDatasetImporter::num_cameras)
        .def("width", &rift::VCIDatasetImporter::width)
        .def("height", &rift::VCIDatasetImporter::height)
        .def("num_frames", &rift::VCIDatasetImporter::num_frames)
        .def("floor_offset", &rift::VCIDatasetImporter::floor_offset)
        .def("start_frame", &rift::VCIDatasetImporter::start_frame)
        .def("flip_images", &rift::VCIDatasetImporter::flip_images)
        .def("flip_masks", &rift::VCIDatasetImporter::flip_masks)
        .def("partial_masks", &rift::VCIDatasetImporter::partial_masks)
        .def("level", &rift::VCIDatasetImporter::level)
        .def("getCameras",
             [](const std::shared_ptr<rift::VCIDatasetImporter>& importer)
             {
                 pybind11::dict data = detail::getCameras(importer);
                 return data;
             });

    m_vcireal_importer.def(py::init<const rift::DatasetHeader&>())
        .def("getMasks",
             (std::vector<torch::Tensor> (rift::VCIRealDatasetImporter::*)(
                 const torch::Tensor&))&rift::VCIRealDatasetImporter::getMasks)
        .def("getMasks",
             (std::vector<torch::Tensor> (rift::VCIRealDatasetImporter::*)(
                 const uint32_t,
                 const torch::Tensor&))&rift::VCIRealDatasetImporter::getMasks)
        .def("getImages", &rift::VCIRealDatasetImporter::getImages)
        .def("getViewProjectionTensor", &rift::VCIRealDatasetImporter::getViewProjectionTensor)
        .def("num_cameras", &rift::VCIRealDatasetImporter::num_cameras)
        .def("width", &rift::VCIRealDatasetImporter::width)
        .def("height", &rift::VCIRealDatasetImporter::height)
        .def("num_frames", &rift::VCIRealDatasetImporter::num_frames)
        .def("floor_offset", &rift::VCIRealDatasetImporter::floor_offset)
        .def("start_frame", &rift::VCIRealDatasetImporter::start_frame)
        .def("flip_images", &rift::VCIRealDatasetImporter::flip_images)
        .def("flip_masks", &rift::VCIRealDatasetImporter::flip_masks)
        .def("partial_masks", &rift::VCIRealDatasetImporter::partial_masks)
        .def("level", &rift::VCIRealDatasetImporter::level)
        .def("getCameras",
             [](const std::shared_ptr<rift::VCIRealDatasetImporter>& importer)
             {
                 pybind11::dict data = detail::getCameras(importer);
                 return data;
             });

    m_vcirealtcp_importer.def(py::init<const rift::DatasetHeader&>())
        .def("getMasks",
             (std::vector<torch::Tensor> (rift::VCIRealTCPDatasetImporter::*)(
                 const torch::Tensor&))&rift::VCIRealTCPDatasetImporter::getMasks)
        .def("getMasks",
             (std::vector<torch::Tensor> (rift::VCIRealTCPDatasetImporter::*)(
                 const uint32_t,
                 const torch::Tensor&))&rift::VCIRealTCPDatasetImporter::getMasks)
        .def("getImages", &rift::VCIRealTCPDatasetImporter::getImages)
        .def("getViewProjectionTensor", &rift::VCIRealTCPDatasetImporter::getViewProjectionTensor)
        .def("num_cameras", &rift::VCIRealTCPDatasetImporter::num_cameras)
        .def("width", &rift::VCIRealTCPDatasetImporter::width)
        .def("height", &rift::VCIRealTCPDatasetImporter::height)
        .def("num_frames", &rift::VCIRealTCPDatasetImporter::num_frames)
        .def("floor_offset", &rift::VCIRealTCPDatasetImporter::floor_offset)
        .def("start_frame", &rift::VCIRealTCPDatasetImporter::start_frame)
        .def("flip_images", &rift::VCIRealTCPDatasetImporter::flip_images)
        .def("flip_masks", &rift::VCIRealTCPDatasetImporter::flip_masks)
        .def("partial_masks", &rift::VCIRealTCPDatasetImporter::partial_masks)
        .def("level", &rift::VCIRealTCPDatasetImporter::level)
        .def("getCameras",
             [](const std::shared_ptr<rift::VCIRealTCPDatasetImporter>& importer)
             {
                 pybind11::dict data = detail::getCameras(importer);
                 return data;
             });
    m_panoptic_importer.def(py::init<const rift::DatasetHeader&>())
        .def("getMasks",
             (std::vector<torch::Tensor> (rift::PanopticDatasetImporter::*)(
                 const torch::Tensor&))&rift::PanopticDatasetImporter::getMasks)
        .def("getMasks",
             (std::vector<torch::Tensor> (rift::PanopticDatasetImporter::*)(
                 const uint32_t,
                 const torch::Tensor&))&rift::PanopticDatasetImporter::getMasks)
        .def("getImages", &rift::PanopticDatasetImporter::getImages)
        .def("getViewProjectionTensor", &rift::PanopticDatasetImporter::getViewProjectionTensor)
        .def("num_cameras", &rift::PanopticDatasetImporter::num_cameras)
        .def("width", &rift::PanopticDatasetImporter::width)
        .def("height", &rift::PanopticDatasetImporter::height)
        .def("num_frames", &rift::PanopticDatasetImporter::num_frames)
        .def("floor_offset", &rift::PanopticDatasetImporter::floor_offset)
        .def("start_frame", &rift::PanopticDatasetImporter::start_frame)
        .def("flip_images", &rift::PanopticDatasetImporter::flip_images)
        .def("flip_masks", &rift::PanopticDatasetImporter::flip_masks)
        .def("partial_masks", &rift::PanopticDatasetImporter::partial_masks)
        .def("level", &rift::PanopticDatasetImporter::level)
        .def("getCameras",
             [](const std::shared_ptr<rift::PanopticDatasetImporter>& importer)
             {
                 pybind11::dict data = detail::getCameras(importer);
                 return data;
             });

    m_tcp_importer.def(py::init<const rift::DatasetHeader&>())
        .def("getMasks",
             (std::vector<torch::Tensor> (rift::TCPDatasetImporter::*)(
                 const torch::Tensor&))&rift::TCPDatasetImporter::getMasks)
        .def("getMasks",
             (std::vector<torch::Tensor> (
                 rift::TCPDatasetImporter::*)(const uint32_t, const torch::Tensor&))&rift::TCPDatasetImporter::getMasks)
        .def("getImages", &rift::TCPDatasetImporter::getImages)
        .def("getViewProjectionTensor", &rift::TCPDatasetImporter::getViewProjectionTensor)
        .def("num_cameras", &rift::TCPDatasetImporter::num_cameras)
        .def("width", &rift::TCPDatasetImporter::width)
        .def("height", &rift::TCPDatasetImporter::height)
        .def("num_frames", &rift::TCPDatasetImporter::num_frames)
        .def("floor_offset", &rift::TCPDatasetImporter::floor_offset)
        .def("start_frame", &rift::TCPDatasetImporter::start_frame)
        .def("flip_images", &rift::TCPDatasetImporter::flip_images)
        .def("flip_masks", &rift::TCPDatasetImporter::flip_masks)
        .def("partial_masks", &rift::TCPDatasetImporter::partial_masks)
        .def("level", &rift::TCPDatasetImporter::level)
        .def("getCameras",
             [](const std::shared_ptr<rift::TCPDatasetImporter>& importer)
             {
                 pybind11::dict data = detail::getCameras(importer);
                 return data;
             });

    m_protocol.attr("VCIPROTOCOL_VERSION_MAJOR") = VCIPROTOCOL_VERSION_MAJOR;
    m_protocol.attr("VCIPROTOCOL_VERSION_MINOR") = VCIPROTOCOL_VERSION_MINOR;

    m_message_task.value("CONNECT_CLIENT", rift::protocol::MessageTask::CONNECT_CLIENT)
        .value("DISCONNECT_CLIENT", rift::protocol::MessageTask::DISCONNECT_CLIENT)
        .value("REQUEST_IMAGE", rift::protocol::MessageTask::REQUEST_IMAGE)
        .value("NO_UPDATE", rift::protocol::MessageTask::NO_UPDATE)
        .value("UPDATE", rift::protocol::MessageTask::UPDATE)
        .value("INVALID", rift::protocol::MessageTask::INVALID)
        .export_values();

    m_protocolversion.def(py::init<>())
        .def("isCompatible", &rift::protocol::ProtocolVersion::isCompatible)
        .def_readonly("major", &rift::protocol::ProtocolVersion::major)
        .def_readonly("minor", &rift::protocol::ProtocolVersion::minor);

    m_protocolheader.def(py::init<>())
        .def(py::init<const rift::protocol::MessageTask>())
        .def_readonly("version", &rift::protocol::ProtocolHeader::version)
        .def_readwrite("task", &rift::protocol::ProtocolHeader::task);

    m_protocol.def("createConnectionMessage", &rift::protocol::createConnectionMessage);
    m_protocol.def("createDisconnectionMessage", &rift::protocol::createDisconnectionMessage);
    m_protocol.def(
        "createRenderRequest",
        [](const uint32_t& width, const uint32_t& height, const torch::Tensor& view, const torch::Tensor& projection)
        {
            auto view_       = view.cpu().clone();
            auto projection_ = view.cpu().clone();

            return rift::protocol::createRenderRequest(width,
                                                       height,
                                                       glm::make_mat4(view_.data_ptr<float>()),
                                                       glm::make_mat4(projection_.data_ptr<float>()));
        });
    m_protocol.def("createNoUpdateMessage", &rift::protocol::createNoUpdateMessage);
    m_protocol.def("createUpdateMessage",
                   [](const torch::Tensor& inv_view_projection,
                      const torch::Tensor& encoded_jpeg,
                      const torch::Tensor& encoded_depth)
                   {
                       auto inv_view_projection_ = inv_view_projection.cpu().clone();
                       return rift::protocol::createUpdateMessage(
                           glm::make_mat4(inv_view_projection_.data_ptr<float>()),
                           encoded_jpeg,
                           encoded_depth);
                   });
}