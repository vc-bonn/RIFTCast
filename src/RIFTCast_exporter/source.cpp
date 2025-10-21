#define ATCG_BUILD_VR true

#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <algorithm>

#include <random>

#include <riftcast/DatasetImporter.h>

#include <portable-file-dialogs.h>

#include <openvr.h>

#include <riftcast/GeometryModule.h>
#include <riftcast/RenderModule.h>
#include <riftcast/InpaintingModule.h>

#include <torch/torch.h>

#include <glad/glad.h>

class RIFTCast : public atcg::Application
{
public:
    RIFTCast(const atcg::WindowProps& props) : atcg::Application(props) {}

    ~RIFTCast()
    {
        geometry_module.reset();
        render_module.reset();
        atcg::Renderer::use();
    }

    struct CMDArguments
    {
        std::string dataset_config;
        std::string output_path;
        bool exclude_nearest         = false;
        bool inpaint                 = true;
        glm::vec4 background_color   = glm::vec4(1);
        std::string calibration_file = "";
        std::string test_file        = "";
        bool test_only               = false;
        int frame                    = -1;
    };

    void printUsage()
    {
        ATCG_INFO("./RIFTCast_exporter --dataset <dataset_path> --output <output_path>");
        ATCG_INFO("Options:");
        ATCG_INFO("\t --exclude_nearest {true | false} \t If the nearest camera should be excluded for texture "
                  "reconstruction (Default: false)");
        ATCG_INFO("\t --inpaint {true | false} \t If inpainting should be performed (linux only) (Default: true)");
        ATCG_INFO("\t --test_only {true | false} \t If only test views should be rendered (Default: false)");
        ATCG_INFO("\t --background [r,g,b,a] \t Background color (Default: [1,1,1,1])");
        ATCG_INFO("\t --calibration <path> \t Override the calibration file (optional)");
        ATCG_INFO("\t --test_file <path> \t Define a set of test cameras (optional)");
        ATCG_INFO("\t --frame <int> \t Only render a specific frame (optional)");
    }

    bool parseColor(const std::string& expr, glm::vec4& color)
    {    // Check for minimum length "(0,0,0,0)"
        if(expr.size() < 9 || expr.front() != '[' || expr.back() != ']')
        {
            return false;
        }

        std::istringstream ss(expr.substr(1, expr.size() - 2));    // Remove parentheses
        float r, g, b, a;
        char comma1, comma2, comma3;

        if(!(ss >> r >> comma1 >> g >> comma2 >> b >> comma3 >> a))
        {
            return false;
        }

        // Ensure commas are present and correct
        if(comma1 != ',' || comma2 != ',' || comma3 != ',')
        {
            return false;
        }

        // Ensure there are no extra characters after parsing
        char extra;
        if(ss >> extra)
        {
            return false;
        }

        color = glm::vec4(r, g, b, a);
        return true;
    }

    bool parseCMDArguments(const std::vector<std::string>& list, CMDArguments& arguments)
    {
        bool success = true;
        std::string invalid_argument;

        if(list.size() < 2)
        {
            ATCG_ERROR("Invalid Usage");
            printUsage();
            return false;
        }

        for(int i = 1; i < list.size(); ++i)
        {
            if(list[i].substr(0, 2) == "--")
            {
                if(i != list.size() - 1)
                {
                    if(list[i].substr(2) == "dataset")
                    {
                        arguments.dataset_config = list[++i];
                        continue;
                    }

                    if(list[i].substr(2) == "output")
                    {
                        arguments.output_path = list[++i];
                        continue;
                    }

                    if(list[i].substr(2) == "exclude_nearest")
                    {
                        if(list[i + 1] == "false")
                        {
                            arguments.exclude_nearest = false;
                            ++i;
                            continue;
                        }
                        else if(list[i + 1] == "true")
                        {
                            arguments.exclude_nearest = true;
                            ++i;
                            continue;
                        }
                    }

                    if(list[i].substr(2) == "inpaint")
                    {
                        if(list[i + 1] == "false")
                        {
                            arguments.inpaint = false;
                            ++i;
                            continue;
                        }
                        else if(list[i + 1] == "true")
                        {
                            arguments.inpaint = true;
                            ++i;
                            continue;
                        }
                    }

                    if(list[i].substr(2) == "test_only")
                    {
                        if(list[i + 1] == "false")
                        {
                            arguments.test_only = false;
                            ++i;
                            continue;
                        }
                        else if(list[i + 1] == "true")
                        {
                            arguments.test_only = true;
                            ++i;
                            continue;
                        }
                    }

                    if(list[i].substr(2) == "background")
                    {
                        bool color_success = parseColor(list[++i], arguments.background_color);
                        if(color_success) continue;
                    }

                    if(list[i].substr(2) == "calibration")
                    {
                        arguments.calibration_file = list[++i];
                        continue;
                    }

                    if(list[i].substr(2) == "test_file")
                    {
                        arguments.test_file = list[++i];
                        continue;
                    }

                    if(list[i].substr(2) == "frame")
                    {
                        arguments.frame = std::stoi(list[++i]);
                        continue;
                    }
                }
            }

            invalid_argument = list[i];
            success          = false;
            break;
        }

        if(!success)
        {
            ATCG_ERROR("Invalid argument: {}", invalid_argument);
            printUsage();
        }

        return success;
    }

    // This is run at the start of the program
    virtual void run() override
    {
        auto cmd_args = atcg::getCommandLineArguments();

        CMDArguments arguments;
        bool parse_success = parseCMDArguments(cmd_args, arguments);

        if(!parse_success) return;

        std::string meta_file  = arguments.dataset_config;
        std::string output_dir = arguments.output_path;


        rift::DatasetHeader header = rift::IO::readDatasetHeader(meta_file);
        if(arguments.frame != -1)
        {
            header.frame_count = 1;
            header.start_frame = arguments.frame;
        }

        if(arguments.calibration_file != "")
        {
            header.camera_path = arguments.calibration_file;
        }

        dataloader = rift::createDatasetImporter(header);

        if(arguments.test_file != "")
        {
            header.camera_path = arguments.test_file;
            dataloader_test    = rift::createDatasetImporter(header);
        }

        geometry_module = atcg::make_ref<rift::GeometryModule>();
        geometry_module->init(0, dataloader);

        render_module = atcg::make_ref<rift::RenderModule>();
        render_module->init(0, dataloader, atcg::JPEGBackend::SOFTWARE);
        render_module->setBackgroundColor(arguments.background_color);

        inpainting_module = atcg::make_ref<rift::InpaintingModule>(dataloader->inpainting_path());

        atcg::Renderer::use();

        cam_valid = torch::ones({dataloader->num_cameras()}, atcg::TensorOptions::int32DeviceOptions());
        atcg::TransformComponent transform;
        transform.setScale(glm::vec3(header.volume_scale));
        transform.setPosition(header.volume_position);
        model = transform.getModel();

        atcg::WorkerPool pool(32);
        pool.start();

        torch::Tensor used_cameras = torch::zeros({(int)dataloader->num_frames(), (int)dataloader->num_cameras(), 3},
                                                  atcg::TensorOptions::int32HostOptions());

        std::filesystem::create_directories(output_dir);
        const std::vector<rift::CameraData>& cameras = dataloader->getCameras();
        std::vector<rift::CameraData> test_cameras;
        if(dataloader_test)
        {
            test_cameras = dataloader_test->getCameras();
        }

        std::vector<float> rec_timings;
        std::vector<float> render_timings;

        for(int frame_idx = dataloader->start_frame(); frame_idx < dataloader->start_frame() + dataloader->num_frames();
            ++frame_idx)
        {
            std::stringstream frame_folder;
            frame_folder << "frame_" << std::setfill('0') << std::setw(5) << frame_idx;
            std::string frame_folder_str = frame_folder.str();
            if(!arguments.test_only)
            {
                std::filesystem::create_directories(output_dir + "/" + frame_folder_str + "/rgb");
                std::filesystem::create_directories(output_dir + "/" + frame_folder_str + "/error");
                std::filesystem::create_directories(output_dir + "/" + frame_folder_str + "/mask");
                std::filesystem::create_directories(output_dir + "/" + frame_folder_str + "/primitives");
                std::filesystem::create_directories(output_dir + "/" + frame_folder_str + "/depth");

                // Visual hull only needs to be computed once per frame
                auto reconstruction = geometry_module->compute_geometry(model, cam_valid, frame_idx);

                auto vertices = reconstruction.vertices;
                auto faces    = reconstruction.faces;
                pool.pushJob(
                    [vertices, faces, frame_folder_str, output_dir]()
                    {
                        atcg::IO::dumpBinary(output_dir + "/" + frame_folder_str + "/vertices.bin", vertices);
                        atcg::IO::dumpBinary(output_dir + "/" + frame_folder_str + "/faces.bin", faces);
                    });

                auto primitive_buffers = geometry_module->getPrimitiveBuffers();

                for(int camera_idx = 0; camera_idx < dataloader->num_cameras(); ++camera_idx)
                {
                    auto primitive_buffer = primitive_buffers[camera_idx]->getColorAttachement(0)->getData(atcg::GPU);
                    // Store primitive map
                    pool.pushJob(
                        [primitive_buffer, frame_folder_str, camera_idx, &cameras, output_dir]()
                        {
                            // atcg::Image img(primitive_buffer.to(torch::kFloat32).cpu());
                            // img.store(output_dir + "/" + frame_folder_str + "/primitives/primitives_" +
                            //           std::to_string(cameras[camera_idx].id) + ".hdr");
                            atcg::IO::dumpBinary(output_dir + "/" + frame_folder_str + "/primitives/primitives_" +
                                                     std::to_string(cameras[camera_idx].id) + ".bin",
                                                 primitive_buffer);
                        });

                    // Just set visible primitives of current camera to 0 -> should be removed from the optimization
                    torch::Tensor visible_copy = reconstruction.visible_primitives.clone();
                    if(arguments.exclude_nearest)
                    {
                        reconstruction.visible_primitives.index_put_({(int)camera_idx, torch::indexing::Slice()}, 0);
                    }
                    render_module->updateState(reconstruction,
                                               cameras[camera_idx].cam,
                                               cameras[camera_idx].width,
                                               cameras[camera_idx].height);
                    auto frame = render_module->renderFrame(cameras[camera_idx].cam);

                    auto color      = frame->getColorAttachement(0)->getData(atcg::GPU);
                    auto visibility = frame->getColorAttachement(2)->getData(atcg::GPU);
                    auto depth      = frame->getColorAttachement(3)->getData(atcg::GPU);

                    auto error = torch::where(visibility == 0, uint8_t(255), uint8_t(0)).to(torch::kUInt8);
                    auto mask  = torch::where(visibility >= 0, uint8_t(255), uint8_t(0)).to(torch::kUInt8);

                    if(arguments.inpaint)
                    {
                        color = inpainting_module->inpaint(color, error.to(torch::kFloat32) / 255.);
                    }

                    auto chosen_cams =
                        torch::arange((int)dataloader->num_cameras(), atcg::TensorOptions::int32HostOptions())
                            .index({render_module->getChosenCameraIndices().to(atcg::CPU) == 1});

                    used_cameras[frame_idx][camera_idx][0] = chosen_cams[0];
                    used_cameras[frame_idx][camera_idx][1] = chosen_cams[1];
                    used_cameras[frame_idx][camera_idx][2] = chosen_cams[2];

                    reconstruction.visible_primitives = visible_copy;

                    pool.pushJob(
                        [color, frame_folder_str, camera_idx, &cameras, output_dir]()
                        {
                            atcg::Image img(color.cpu());
                            img.store(output_dir + "/" + frame_folder_str + "/rgb/rgb_" +
                                      std::to_string(cameras[camera_idx].id) + ".png");
                        });

                    pool.pushJob(
                        [error, frame_folder_str, camera_idx, &cameras, output_dir]()
                        {
                            atcg::Image img(error.cpu());
                            img.store(output_dir + "/" + frame_folder_str + "/error/error_" +
                                      std::to_string(cameras[camera_idx].id) + ".png");
                        });

                    pool.pushJob(
                        [mask, frame_folder_str, camera_idx, &cameras, output_dir]()
                        {
                            atcg::Image img(mask.cpu());
                            img.store(output_dir + "/" + frame_folder_str + "/mask/mask_" +
                                      std::to_string(cameras[camera_idx].id) + ".png");
                        });

                    pool.pushJob(
                        [depth, frame_folder_str, camera_idx, &cameras, output_dir]
                        {
                            atcg::IO::dumpBinary(output_dir + "/" + frame_folder_str + "/depth/depth_" +
                                                     std::to_string(cameras[camera_idx].id) + ".bin",
                                                 depth.cpu());
                            // float n    = 0.01f;
                            // float f    = 1000.0f;
                            // depth      = (2.0f * n * f) / (f + n - (depth * 2.0f - 1.0f) * (f - n));
                            // atcg::Image img(depth);
                            // img.store(frame_dir + "/depth/depth_" + std::to_string(camera.id) + ".hdr");
                        });
                }
            }

            if(dataloader_test)
            {
                std::filesystem::create_directories(output_dir + "/" + frame_folder_str + "/test");

                // Warmup
                for(int i = 0; i < 10; ++i)
                {
                    auto reconstruction = geometry_module->compute_geometry(model, cam_valid, frame_idx);
                    render_module->updateState(reconstruction,
                                               test_cameras[0].cam,
                                               test_cameras[0].width,
                                               test_cameras[0].height);

                    render_module->renderFrame(test_cameras[0].cam);
                }

                for(int camera_idx = 0; camera_idx < dataloader_test->num_cameras(); ++camera_idx)
                {
                    atcg::ref_ptr<atcg::Framebuffer> frame;

                    // Do reconstruction every frame for better comparison
                    torch::cuda::synchronize();
                    glFlush();
                    atcg::Timer timer_reconstruction;
                    auto reconstruction = geometry_module->compute_geometry(model, cam_valid, frame_idx);
                    torch::cuda::synchronize();
                    glFlush();
                    rec_timings.push_back(timer_reconstruction.elapsedMillis());

                    // Measure rendering
                    {
                        torch::cuda::synchronize();
                        glFlush();
                        atcg::Timer timer_rendering;
                        render_module->updateState(reconstruction,
                                                   test_cameras[camera_idx].cam,
                                                   test_cameras[camera_idx].width,
                                                   test_cameras[camera_idx].height);

                        frame = render_module->renderFrame(test_cameras[camera_idx].cam);
                        torch::cuda::synchronize();
                        glFlush();
                        render_timings.push_back(timer_rendering.elapsedMillis());
                    }

                    auto color      = frame->getColorAttachement(0)->getData(atcg::GPU);
                    auto visibility = frame->getColorAttachement(2)->getData(atcg::GPU);

                    auto error = torch::where(visibility == 0, uint8_t(255), uint8_t(0)).to(torch::kUInt8);
                    auto mask  = torch::where(visibility >= 0, uint8_t(255), uint8_t(0)).to(torch::kUInt8);

                    if(arguments.inpaint)
                    {
                        color = inpainting_module->inpaint(color, error.to(torch::kFloat32) / 255.);
                    }

                    pool.pushJob(
                        [color, frame_folder_str, camera_idx, &test_cameras, output_dir]()
                        {
                            atcg::Image img(color.cpu());
                            img.store(output_dir + "/" + frame_folder_str + "/test/rgb_" +
                                      std::to_string(test_cameras[camera_idx].id) + ".png");
                        });
                }
            }
        }

        atcg::IO::dumpBinary(output_dir + "/used_cameras.bin", used_cameras);
        pool.waitDone();

        torch::Tensor rec_tensor =
            atcg::createHostTensorFromPointer<float>(rec_timings.data(), {(int)rec_timings.size()});
        torch::Tensor render_tensor =
            atcg::createHostTensorFromPointer<float>(render_timings.data(), {(int)render_timings.size()});

        atcg::IO::dumpBinary(output_dir + "/timings_reconstruction.bin", rec_tensor);
        atcg::IO::dumpBinary(output_dir + "/timings_rendering.bin", render_tensor);
    }

private:
    float time_passed                                 = 0.0f;
    float reconstruction_time                         = 0.0f;
    float mapping_time                                = 0.0f;
    float synchronizing_time                          = 0.0f;
    float display_time                                = 0.0f;
    atcg::CyclicCollection<float> time_collection     = atcg::CyclicCollection<float>("Time Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> fps_collection      = atcg::CyclicCollection<float>("FPS Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> fps_collection_mean = atcg::CyclicCollection<float>("FPS Collection", 5);
    atcg::CyclicCollection<float> reconstruction_collection =
        atcg::CyclicCollection<float>("Reconstruction Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> reconstruction_collection_mean =
        atcg::CyclicCollection<float>("Reconstruction Collection", 5);
    atcg::CyclicCollection<float> synchronizing_collection =
        atcg::CyclicCollection<float>("Synchronizing Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> synchronizing_collection_mean =
        atcg::CyclicCollection<float>("Synchronizing Collection", 5);
    atcg::CyclicCollection<float> mapping_collection = atcg::CyclicCollection<float>("Mapping Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> mapping_collection_mean = atcg::CyclicCollection<float>("Mapping Collection", 5);

    atcg::ref_ptr<atcg::Scene> scene;
    atcg::Entity hovered_entity;
    atcg::Entity mesh_entity;

    atcg::ref_ptr<rift::DatasetImporter> dataloader;
    atcg::ref_ptr<rift::DatasetImporter> dataloader_test;
    atcg::ref_ptr<rift::GeometryModule> geometry_module;
    atcg::ref_ptr<rift::RenderModule> render_module;
    atcg::ref_ptr<rift::InpaintingModule> inpainting_module;
    torch::Tensor cam_valid;
    glm::mat4 model;

    atcg::ref_ptr<atcg::CameraController> camera_controller;

    bool in_viewport = false;

    bool hover_mode = false;

    glm::vec2 mouse_pos;

    bool show_render_settings = false;
#ifndef ATCG_HEADLESS
    ImGuizmo::OPERATION current_operation = ImGuizmo::OPERATION::TRANSLATE;
#endif

private:
};

atcg::Application* atcg::createApplication()
{
    atcg::WindowProps props;
    props.vsync  = true;
    props.hidden = true;
    return new RIFTCast(props);
}