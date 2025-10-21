#define BUILD_VR true

#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <algorithm>

#include <random>
#include <stb_image.h>

#include <SFML/Network.hpp>

#include <zstd.h>

#include <riftcast/riftcastkernels.h>

#include <riftcast/protocol/riftcastprotocol.h>
#include <riftcast/BenchmarkLogger.h>

class RIFTCastClientLayer : public atcg::Layer
{
public:
    RIFTCastClientLayer(const std::string& name) : atcg::Layer(name) {}

    ~RIFTCastClientLayer()
    {
        running = false;
        if(request_thread.joinable()) request_thread.join();

        client.disconnect();
    }

    torch::Tensor
    request_vertices(const glm::mat4& view, const glm::mat4& projection, const uint32_t width, const uint32_t height)
    {
        atcg::Timer timer;
        // Send camera data
        auto message = rift::protocol::createRenderRequest(width, height, view, projection);

        auto received_data = client.sendAndWait(message);

        uint32_t offset = 0;
        rift::protocol::ProtocolHeader header =
            atcg::NetworkUtils::readStruct<rift::protocol::ProtocolHeader>(received_data.data(), offset);

        if(!header.version.isCompatible())
        {
            ATCG_ERROR("Protocols not compatible. This version of the protocol is {}.{}. Received message had version "
                       "{}.{}",
                       RIFTCAST_PROTOCOL_VERSION_MAJOR,
                       RIFTCAST_PROTOCOL_VERSION_MINOR,
                       header.version.major,
                       header.version.minor);
            return {};
        }

        if(header.task == rift::protocol::MessageTask::NO_UPDATE) return {};

        bandwidth_logger.logSample(received_data.size());

        uint32_t projection_size = atcg::NetworkUtils::readInt<uint32_t>(received_data.data(), offset);
        ATCG_ASSERT(projection_size == sizeof(glm::mat4), "Received wrong size for projection");
        float* projection_data = (float*)(received_data.data() + offset);
        offset += projection_size;
        glm::mat4 inv_view_projection = glm::make_mat4(projection_data);

        uint32_t image_size = atcg::NetworkUtils::readInt<uint32_t>(received_data.data(), offset);

        std::vector<std::vector<uint8_t>> file_data(1);

        file_data[0].resize(image_size);
        std::memcpy(file_data[0].data(), received_data.data() + offset, image_size);

        offset += image_size;

        uint32_t depth_size = atcg::NetworkUtils::readInt<uint32_t>(received_data.data(), offset);

        auto img = decoder->decompressImages(file_data).squeeze(0);

        auto depth_data = torch::empty({img.size(0), img.size(1)}, atcg::TensorOptions::int16HostOptions());
        ZSTD_decompress(depth_data.data_ptr(),
                        depth_data.element_size() * depth_data.numel(),
                        received_data.data() + offset,
                        depth_size);

        auto vertices = rift::unprojectVerticesQuantized(img, inv_view_projection, depth_data.cuda());

        runtime_logger.logSample(timer.elapsedMillis());

        return vertices;
    }

    void request_data()
    {
        while(running)
        {
            if(!done)
            {
                current_vertices = request_vertices(current_view, current_projection, current_width, current_height);
                done             = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    torch::Tensor get_draw_data(const atcg::ref_ptr<atcg::Camera>& camera, const uint32_t width, const uint32_t height)
    {
        if(stop_request)
        {
            return last_vertices;
        }

        if(done)
        {
            // atcg::IO::dumpBinary("vertices.bin", last_vertices);
            // auto entity     = scene->getEntitiesByName("Model").front();
            // glm::mat4 model = entity.getComponent<atcg::TransformComponent>();
            current_view = camera->getView();    //* model;

            // The server should render a larger frustum so we don't get screen space rendering artifacts
            glm::mat4 transform = glm::scale(glm::vec3(1.0f / scaling_factor, 1.0f / scaling_factor, 1.0f));
            current_projection  = transform * camera->getProjection();

            current_width  = width;
            current_height = height;

            if(current_vertices.numel() > 0) last_vertices = current_vertices.clone();
            done = false;
        }

        return last_vertices;
    }

    struct CMDArguments
    {
        bool skybox   = true;
        bool cameras  = true;
        bool cad_grid = true;
    };

    void printUsage()
    {
        ATCG_INFO("./RIFTCast_client");
        ATCG_INFO("Options:");
        ATCG_INFO("\t --skybox {true|false} \t If a skybox should be loaded (Default: true)");
        ATCG_INFO("\t --cameras {true|false} \t If the cameras should be rendered (Default: true)");
        ATCG_INFO("\t --cad_grid {true|false} \t If the cad grid should be rendered (Default: true)");
    }

    bool parseCMDArguments(const std::vector<std::string>& list, CMDArguments& arguments)
    {
        bool success = true;
        std::string invalid_argument;

        for(int i = 1; i < list.size(); ++i)
        {
            if(list[i].substr(0, 2) == "--")
            {
                if(i != list.size() - 1)
                {
                    if(list[i].substr(2) == "skybox")
                    {
                        if(list[i + 1] == "false")
                        {
                            arguments.skybox = false;
                            ++i;
                            continue;
                        }
                        else if(list[i + 1] == "true")
                        {
                            arguments.skybox = true;
                            ++i;
                            continue;
                        }
                    }

                    if(list[i].substr(2) == "cameras")
                    {
                        if(list[i + 1] == "false")
                        {
                            arguments.cameras = false;
                            ++i;
                            continue;
                        }
                        else if(list[i + 1] == "true")
                        {
                            arguments.cameras = true;
                            ++i;
                            continue;
                        }
                    }

                    if(list[i].substr(2) == "cad_grid")
                    {
                        if(list[i + 1] == "false")
                        {
                            arguments.cad_grid = false;
                            ++i;
                            continue;
                        }
                        else if(list[i + 1] == "true")
                        {
                            arguments.cad_grid = true;
                            ++i;
                            continue;
                        }
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
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(false);
        atcg::Renderer::setClearColor(glm::vec4(1));

        auto t  = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream base;

        base << "bin/log/" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");

        bandwidth_logger = rift::BenchmarkLogger("bandwidth", base.str() + "_bandwidth.txt");
        runtime_logger   = rift::BenchmarkLogger("runtime", base.str() + "_runtime.txt");
        display_logger   = rift::BenchmarkLogger("display", base.str() + "_display.txt");

        auto cmd_args = atcg::getCommandLineArguments();

        CMDArguments arguments;
        bool parse_success = parseCMDArguments(cmd_args, arguments);

        if(!parse_success) return;

        scene = atcg::make_ref<atcg::Scene>();

        if(arguments.skybox)
        {
            auto skybox = atcg::IO::imread("res/skybox_vci.hdr");
            scene->setSkybox(skybox);
        }

        uint32_t width, height;
        if(BUILD_VR && atcg::VR::isVRAvailable())
        {
            atcg::VR::setNear(0.01f);
            atcg::VR::setFar(10.0f);
            width              = atcg::VR::width();
            height             = atcg::VR::height();
            float aspect_ratio = (float)width / (float)height;
            atcg::CameraIntrinsics instrinsics_left(atcg::VR::getProjection(atcg::VRSystem::Eye::LEFT));
            atcg::CameraIntrinsics instrinsics_right(atcg::VR::getProjection(atcg::VRSystem::Eye::RIGHT));
            instrinsics_left.setAspectRatio(aspect_ratio);
            instrinsics_right.setAspectRatio(aspect_ratio);

            atcg::CameraExtrinsics extrinsics_left(glm::inverse(atcg::VR::getInverseView(atcg::VRSystem::Eye::LEFT)));
            atcg::CameraExtrinsics extrinsics_right(glm::inverse(atcg::VR::getInverseView(atcg::VRSystem::Eye::RIGHT)));
            camera_controller = atcg::make_ref<atcg::VRController>(
                atcg::make_ref<atcg::PerspectiveCamera>(extrinsics_left, instrinsics_left),
                atcg::make_ref<atcg::PerspectiveCamera>(extrinsics_right, instrinsics_right));
            atcg::VR::initControllerMeshes(scene);
        }
        else
        {
            const auto& window = atcg::Application::get()->getWindow();
            width              = window->getWidth();
            height             = window->getHeight();
            float aspect_ratio = (float)width / (float)height;
            atcg::CameraExtrinsics extrinsics;
            atcg::CameraIntrinsics intrinsics;
            intrinsics.setAspectRatio(aspect_ratio);
            camera_controller = atcg::make_ref<atcg::FirstPersonController>(
                atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics));
        }

        auto f = pfd::open_file("Choose scene meta file", pfd::path::home(), {"Json", "*.json"}, pfd::opt::none);
        std::string meta_file = f.result()[0];

        nlohmann::json data;

        try
        {
            std::fstream f(meta_file);
            data = nlohmann::json::parse(f);
        }
        catch(const std::exception& e)
        {
            ATCG_ERROR("Failed to load meta file '{0}'\n   {1}", meta_file.c_str(), e.what());
        }

        std::string ip = data["server"]["ip"];
        int port       = data["server"]["port"];

        client.connect(ip, port);

        target_width  = width * scaling_factor;
        target_height = height * scaling_factor;

        decoder = atcg::make_ref<atcg::JPEGDecoder>(1, (uint32_t)(target_width), (uint32_t)(target_height));

        camera_controller->getCamera()->setFar(10.0f);

        {
            auto entity         = scene->createEntity("Model");
            auto& renderer      = entity.addComponent<atcg::PointRenderComponent>();
            renderer.color      = glm::vec3(1);
            renderer.shader     = atcg::ShaderManager::getShader("flat");
            renderer.point_size = 3.0f;
            auto& transform     = entity.addComponent<atcg::TransformComponent>();
            // transform.setPosition(glm::vec3(0.0f, 1.53f, 2.4f));
            // transform.setRotation(glm::vec3(0, 0, -glm::pi<float>() / 2));
            model_geometry = atcg::Graph::createPointCloud();
            entity.addComponent<atcg::GeometryComponent>(model_geometry);
        }

        current_width  = target_width;
        current_height = target_height;

        if(arguments.cameras)
        {
            auto message       = rift::protocol::createCameraRequest();
            auto received_data = client.sendAndWait(message);

            uint32_t offset           = sizeof(rift::protocol::ProtocolHeader);
            uint32_t view_buffer_size = atcg::NetworkUtils::readInt<uint32_t>(received_data.data(), offset);
            uint32_t num_cameras      = view_buffer_size / sizeof(glm::mat4);

            std::vector<glm::mat4> views(num_cameras);

            std::memcpy(views.data(), received_data.data() + offset, view_buffer_size);
            offset += view_buffer_size;

            uint32_t projection_buffer_size = atcg::NetworkUtils::readInt<uint32_t>(received_data.data(), offset);
            ATCG_ASSERT(num_cameras == projection_buffer_size / sizeof(glm::mat4), "Number of cameras have to be same");

            std::vector<glm::mat4> projections(num_cameras);

            std::memcpy(projections.data(), received_data.data() + offset, projection_buffer_size);

            for(int i = 0; i < num_cameras; ++i)
            {
                auto entity = scene->createEntity("Camera_" + std::to_string(i));
                auto cam    = atcg::make_ref<atcg::PerspectiveCamera>();
                cam->setView(views[i]);
                cam->setProjection(projections[i]);
                auto& camera_component        = entity.addComponent<atcg::CameraComponent>(cam);
                camera_component.render_scale = 0.3f;
            }
        }

        render_cad = arguments.cad_grid;

        running        = true;
        request_thread = std::thread(&RIFTCastClientLayer::request_data, this);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        if(!camera_ride)
        {
            camera_controller->onUpdate(delta_time);
        }

        // auto intermediate_tensor =
        //     torch::cat(
        //         {img, torch::full({img.size(0), img.size(1), 1}, 255, atcg::TensorOptions::uint8DeviceOptions())},
        //         -1)
        //         .contiguous();

        // atcg::Framebuffer::useDefault();
        // atcg::Renderer::getFramebuffer()->getColorAttachement(0)->setData(intermediate_tensor);
        // atcg::Renderer::useScreenBuffer();

        if(BUILD_VR && atcg::VR::isVRAvailable())
        {
            atcg::ref_ptr<atcg::VRController> controller =
                std::dynamic_pointer_cast<atcg::VRController>(camera_controller);

            auto vertices = get_draw_data(controller->getCameraLeft(), target_width, target_height);

            if(vertices.numel() > 0) model_geometry->updateVertices(vertices);

            if(controller->inMovement())
            {
                atcg::VR::setMovementLine(controller->getControllerPosition(), controller->getControllerIntersection());
            }

            auto [t_left, t_right] = atcg::VR::getRenderTargets();

            t_left->use();
            atcg::Renderer::setViewport(0, 0, atcg::VR::width(), atcg::VR::height());

            atcg::Renderer::clear();

            controller->getCameraLeft()->setFar(1000.0f);
            scene->draw(controller->getCameraLeft());
            controller->getCameraLeft()->setFar(10.0f);

            atcg::Renderer::drawCameras(scene, controller->getCameraLeft());

            if(render_cad)
            {
                atcg::Renderer::drawCADGrid(camera_controller->getCamera());
            }

            if(controller->inMovement())
            {
                atcg::VR::drawMovementLine(controller->getCameraLeft());
            }

            t_right->use();

            atcg::Renderer::clear();

            controller->getCameraRight()->setFar(1000.0f);
            scene->draw(controller->getCameraRight());
            controller->getCameraRight()->setFar(10.0f);

            atcg::Renderer::drawCameras(scene, controller->getCameraRight());

            if(render_cad)
            {
                atcg::Renderer::drawCADGrid(camera_controller->getCamera());
            }

            // atcg::Renderer::drawCADGrid(controller->getCameraRight());

            if(controller->inMovement())
            {
                atcg::VR::drawMovementLine(controller->getCameraRight());
            }

            atcg::Renderer::useScreenBuffer();
            atcg::Renderer::setDefaultViewport();

            atcg::VR::renderToScreen();
        }
        else
        {
            if(camera_ride)
            {
                static float angle = 0.0f;

                angle += 0.05f * delta_time;

                float c = glm::cos(atcg::Constants::two_pi<float>() * angle);
                float s = glm::sin(atcg::Constants::two_pi<float>() * angle);
                float r = 2.5f;

                camera_controller->getCamera()->setPosition(glm::vec3(r * c, 1.8f, r * s));
                camera_controller->getCamera()->setLookAt(glm::vec3(0.0f, 1.0f, 0.0f));
            }

            if(camera_swing)
            {
                static float angle = 0.0f;

                angle += 0.4f * delta_time;

                float c = glm::cos(atcg::Constants::two_pi<float>() * angle);
                float s = glm::sin(atcg::Constants::two_pi<float>() * angle);
                float r = 0.12f;

                glm::mat4 inv_V = glm::inverse(camera_controller->getCamera()->getView());
                glm::vec3 left  = inv_V[0];
                glm::vec3 up    = inv_V[1];

                glm::vec3 new_pos = center + r * glm::normalize(c * left + s * up);
                camera_controller->getCamera()->setPosition(new_pos);
                camera_controller->getCamera()->setLookAt(glm::vec3(0.0f, 1.0f, 0.0f));
            }

            auto vertices = get_draw_data(camera_controller->getCamera(), target_width, target_height);

            if(vertices.numel() > 0) model_geometry->updateVertices(vertices);

            atcg::Renderer::clear();
            camera_controller->getCamera()->setFar(1000.0f);
            scene->draw(camera_controller->getCamera());
            camera_controller->getCamera()->setFar(10.0f);
            atcg::Renderer::drawCameras(scene, camera_controller->getCamera());

            if(render_cad)
            {
                atcg::Renderer::drawCADGrid(camera_controller->getCamera());
            }
        }

        display_logger.logSample(delta_time * 1000.0f);
    }

#ifndef ATCG_HEADLESS
    virtual void onImGuiRender() override
    {
        ImGui::Begin("Config");

        ImGui::Checkbox("Stop request", &stop_request);

        ImGui::End();
    }
#endif

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::WindowResizeEvent>(ATCG_BIND_EVENT_FN(RIFTCastClientLayer::onResize));
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(RIFTCastClientLayer::onKeyPress));
    }

    bool onResize(atcg::WindowResizeEvent* event)
    {
        if(!BUILD_VR || !atcg::VR::isVRAvailable())
        {
            target_width  = event->getWidth() * scaling_factor;
            target_height = event->getHeight() * scaling_factor;
            decoder       = atcg::make_ref<atcg::JPEGDecoder>(1, (uint32_t)(target_width), (uint32_t)(target_height));
        }
        return false;
    }

    bool onKeyPress(atcg::KeyPressedEvent* event)
    {
        if(event->getKeyCode() == ATCG_KEY_K)
        {
            camera_ride = !camera_ride;
        }

        if(event->getKeyCode() == ATCG_KEY_O)
        {
            camera_swing = !camera_swing;
            center       = camera_controller->getCamera()->getPosition();
            look_at      = camera_controller->getCamera()->getLookAt();
        }

        if(event->getKeyCode() == ATCG_KEY_T)
        {
            atcg::VR::setOffset(glm::vec3(0));
        }

        return false;
    }

private:
    atcg::ref_ptr<atcg::Scene> scene;
    atcg::ref_ptr<atcg::Graph> model_geometry;

    torch::Tensor current_vertices, last_vertices;
    uint32_t current_width, current_height;
    float scaling_factor = 1.2f;
    glm::mat4 current_view, current_projection;
    std::thread request_thread;
    std::atomic_bool done = false;
    bool stop_request     = false;

    bool camera_ride  = false;
    bool camera_swing = false;
    glm::vec3 center;
    glm::vec3 look_at;

    atcg::ref_ptr<atcg::CameraController> camera_controller;

    atcg::TCPClient client;

    atcg::ref_ptr<atcg::JPEGDecoder> decoder;

    std::atomic_bool running = false;

    uint32_t target_width, target_height;

    rift::BenchmarkLogger bandwidth_logger;
    rift::BenchmarkLogger runtime_logger;
    rift::BenchmarkLogger display_logger;

    atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler> panel;

    bool render_cad = true;
};

class RIFTCastClient : public atcg::Application
{
public:
    RIFTCastClient(const atcg::WindowProps& props) : atcg::Application(props)
    {
        pushLayer(new RIFTCastClientLayer("Layer"));
    }

    ~RIFTCastClient() {}
};

atcg::Application* atcg::createApplication()
{
    atcg::WindowProps props;
    props.vsync = false;
    return new RIFTCastClient(props);
}