#define ATCG_BUILD_VR true

#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <algorithm>

#include <random>

#include <riftcast/DatasetImporter.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>
#include <torch/cuda.h>

#include <portable-file-dialogs.h>

#include <thread>
#include <mutex>

#include <riftcast/GeometryModule.h>
#include <riftcast/RenderModule.h>
#include <riftcast/riftcastkernels.h>

#ifndef ATCG_HEADLESS
    #include <implot.h>
#endif

class RIFTCastLayer : public atcg::Layer
{
public:
    RIFTCastLayer(const std::string& name) : atcg::Layer(name) {}

    void destroy()
    {
        running = false;
        if(visual_hull_thread.joinable()) visual_hull_thread.join();
        if(render_thread.joinable()) render_thread.join();

        // geometry_module.reset();
        // render_module.reset();
        atcg::Renderer::use();
    }

    void visual_hull()
    {
        int visual_hull_device_id = 0;

        at::cuda::CUDAGuard device_guard(visual_hull_device_id);
        auto cam_valid = torch::ones({dataloader->num_cameras()}, atcg::TensorOptions::int32DeviceOptions());

        auto geometry_module = atcg::make_ref<rift::GeometryModule>();
        geometry_module->init(visual_hull_device_id, dataloader);

        float delta_time = 1.0f / 60.0f;
        while(running)
        {
            atcg::Timer timer;
            auto reconstruction = geometry_module->compute_geometry(model, cam_valid);
            {
                std::lock_guard vh_guard(visual_hull_mutex);
                output_vertices   = reconstruction.vertices;
                output_faces      = reconstruction.faces;
                output_normals    = reconstruction.normals;
                output_primitives = reconstruction.visible_primitives;
                current_frame     = dataloader->getLastAvailableFrame();
            }


            delta_time = timer.elapsedSeconds();

            {
                std::lock_guard guard(statistic_mutex);
                current_reconstruction_time = timer.elapsedMillis();
            }
        }

        geometry_module.reset();
    }

    atcg::ref_ptr<rift::RenderModule> render_module;
    void render()
    {
        render_module = atcg::make_ref<rift::RenderModule>();
        render_module->init(0, dataloader, atcg::JPEGBackend::SOFTWARE);

        atcg::CameraExtrinsics extrinsics;
        atcg::CameraIntrinsics intrinsics;
        atcg::ref_ptr<atcg::PerspectiveCamera> camera = atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics);

        atcg::Timer recompile_timer;

        while(true)
        {
            if(!running)
            {
                break;
            }

            if(!start_rendering)
            {
                continue;
            }

            start_rendering = false;

            atcg::Timer timer;

            // 2. Get camera parameters
            uint32_t width  = render_input_width;
            uint32_t height = render_input_height;

            glm::mat4 view       = render_input_view;
            glm::mat4 projection = render_input_projection;
            camera->setView(view);
            camera->setProjection(projection);

            rift::GeometryReconstruction reconstruction;
            {
                std::lock_guard lock(visual_hull_mutex);
                reconstruction.current_frame = current_frame;
                if(output_vertices.numel() > 0 && output_faces.numel() > 0)
                {
                    reconstruction.vertices           = output_vertices.clone();
                    reconstruction.faces              = output_faces.clone();
                    reconstruction.normals            = output_normals.clone();
                    reconstruction.visible_primitives = output_primitives.clone();
                    torch::cuda::synchronize();    // Memcpys are async
                }
            }

            render_module->updateState(reconstruction, camera, width, height);

            auto framebuffer = render_module->renderFrame(camera);

            // 5. Encode image
            auto img_data    = framebuffer->getColorAttachement(0)->getData(atcg::GPU);
            auto depth_data  = framebuffer->getColorAttachement(3)->getData(atcg::GPU);
            auto normal_data = framebuffer->getColorAttachement(4)->getData(atcg::GPU);

            {
                // Send output to inpainting thread
                std::lock_guard guard(render_mutex);
                render_output_img                 = img_data;
                render_output_depth               = depth_data;
                render_output_normals             = normal_data;
                render_output_inv_view_projection = glm::inverse(projection * view);

                rendering_done = true;
            }

            {
                std::lock_guard guard(statistic_mutex);
                current_mapping_time = timer.elapsedMillis();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));

            if(recompile_timer.elapsedSeconds() >= 1.0f)
            {
                render_module->getShaderManager()->onUpdate();
                recompile_timer.reset();
            }
        }

        render_module.reset();
    }

    void updateReconstruction()
    {
        atcg::Timer timer;

        torch::Tensor output_img;
        torch::Tensor output_depth;
        torch::Tensor output_normals;
        glm::mat4 inv_view_projection;
        if(rendering_done)
        {
            rendering_done = false;
            // 1. Update input for rendering thread
            render_input_width      = atcg::Renderer::getFramebuffer()->width();
            render_input_height     = atcg::Renderer::getFramebuffer()->height();
            render_input_view       = camera_controller->getCamera()->getView();
            render_input_projection = camera_controller->getCamera()->getProjection();

            // 2. Get pointcloud data
            if(render_output_img.numel() > 0)    // Should only happen for first frame
            {
                output_img          = render_output_img.clone();
                output_depth        = render_output_depth.clone();
                output_normals      = render_output_normals.clone();
                inv_view_projection = render_output_inv_view_projection;
                torch::cuda::synchronize();
            }


            // 3. Start rendering thread again
            start_rendering = true;
        }

        if(output_img.numel() <= 0) return;

        // Update point cloud
        auto [vertices, colors, normals] =
            rift::unprojectVertices(output_img, inv_view_projection, output_depth, output_normals);
        if(vertices.numel() > 0)
        {
            pointcloud->resizeVertices(vertices.size(0));
            pointcloud->getDevicePositions().index_put_({torch::indexing::Slice(), torch::indexing::Slice()}, vertices);
            pointcloud->getDeviceColors().index_put_({torch::indexing::Slice(), torch::indexing::Slice()}, colors);
            pointcloud->getDeviceNormals().index_put_({torch::indexing::Slice(), torch::indexing::Slice()}, normals);
            pointcloud->unmapDeviceVertexPointer();
        }

        current_synchronizing_time = timer.elapsedMillis();
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(true);
        atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));
        atcg::Renderer::toggleCulling(false);
        atcg::Renderer::toggleMSAA(false);

        const auto& window = atcg::Application::get()->getWindow();
        scene              = atcg::make_ref<atcg::Scene>();

        {
            auto reconstruction = scene->createEntity("Reconstruction");
            reconstruction.addComponent<atcg::TransformComponent>();
            pointcloud = atcg::Graph::createPointCloud();
            reconstruction.addComponent<atcg::GeometryComponent>(pointcloud);
            auto& renderer      = reconstruction.addComponent<atcg::PointRenderComponent>();
            renderer.point_size = 2;
            renderer.shader     = atcg::ShaderManager::getShader("flat");
        }

        if(atcg::VR::isVRAvailable() && ATCG_BUILD_VR)
        {
            atcg::VR::setNear(0.01f);
            atcg::VR::setFar(10.0f);
            float aspect_ratio = (float)atcg::VR::width() / (float)atcg::VR::height();
            atcg::CameraExtrinsics extrinsics;
            atcg::CameraIntrinsics intrinsics;
            intrinsics.setAspectRatio(aspect_ratio);
            camera_controller =
                atcg::make_ref<atcg::VRController>(atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics),
                                                   atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics));
            atcg::VR::initControllerMeshes(scene);
        }
        else
        {
            float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
            atcg::CameraExtrinsics extrinsics;
            atcg::CameraIntrinsics intrinsics;
            intrinsics.setAspectRatio(aspect_ratio);
            camera_controller = atcg::make_ref<atcg::FirstPersonController>(
                atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics));
        }

        // camera_controller->getCamera()->setPosition(glm::vec3(3.0f, 1.0f, 0.0f));
        // camera_controller->getCamera()->setLookAt(glm::vec3(0.0f, 1.0f, 0.0f));

        auto skybox = atcg::IO::imread("res/skybox_vci.hdr");
        scene->setSkybox(skybox);

        auto f = pfd::open_file("Choose scene meta file", pfd::path::home(), {"Json", "*.json"}, pfd::opt::none);
        std::string meta_file = f.result()[0];

        rift::DatasetHeader header = rift::IO::readDatasetHeader(meta_file);
        dataloader                 = rift::createDatasetImporter(header);

        atcg::TransformComponent transform;
        transform.setScale(glm::vec3(header.volume_scale));
        transform.setPosition(header.volume_position);
        model = transform.getModel();

        {
            auto volume = scene->createEntity("Volume");
            volume.addComponent<atcg::TransformComponent>(transform);
            std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1.0f, -1.0f, -1.0f)),
                                                  atcg::Vertex(glm::vec3(-1.0f, -1.0f, 1.0f)),
                                                  atcg::Vertex(glm::vec3(-1.0f, 1.0f, -1.0f)),
                                                  atcg::Vertex(glm::vec3(-1.0f, 1.0f, 1.0f)),
                                                  atcg::Vertex(glm::vec3(1.0f, -1.0f, -1.0f)),
                                                  atcg::Vertex(glm::vec3(1.0f, -1.0f, 1.0f)),
                                                  atcg::Vertex(glm::vec3(1.0f, 1.0f, -1.0f)),
                                                  atcg::Vertex(glm::vec3(1.0f, 1.0f, 1.0f))};

            std::vector<atcg::Edge> edges = {atcg::Edge {glm::vec2(0, 1), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(0, 2), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(0, 4), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(1, 5), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(1, 3), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(2, 6), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(2, 3), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(3, 7), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(6, 4), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(6, 7), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(4, 5), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(5, 7), glm::vec3(1), 1.0f}};

            volume.addComponent<atcg::GeometryComponent>(atcg::Graph::createGraph(vertices, edges));
            volume.addComponent<atcg::EdgeRenderComponent>();
        }

        const auto& cameras = dataloader->getCameras();

        for(int i = 0; i < cameras.size(); ++i)
        {
            atcg::ref_ptr<atcg::PerspectiveCamera> cam = cameras[i].cam;
            auto entity                                = scene->createEntity(cameras[i].name);
            auto& transform                            = entity.addComponent<atcg::TransformComponent>();

            transform.setModel(glm::inverse(cam->getView()) *
                               glm::scale(glm::vec3(1.0f, 1.0f, glm::tan(glm::radians(cam->getFOV()) / 2.0f))));
            auto& component        = entity.addComponent<atcg::CameraComponent>(cam);
            component.render_scale = 0.2f;
            component.width        = cameras[i].width;
            component.height       = cameras[i].height;
        }

        scene->setCamera(camera_controller->getCamera());

        panel = atcg::SceneHierarchyPanel(scene);

        running            = true;
        visual_hull_thread = std::thread(&RIFTCastLayer::visual_hull, this);
        render_thread      = std::thread(&RIFTCastLayer::render, this);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        if(!camera_ride)
        {
            camera_controller->onUpdate(delta_time);
        }

        updateReconstruction();

        atcg::Renderer::setClearColor(glm::vec4(1));
        atcg::Renderer::clear();

        scene->draw(camera_controller->getCamera());
        atcg::Renderer::drawCameras(scene, camera_controller->getCamera());
        atcg::Renderer::drawCADGrid(camera_controller->getCamera());

        time_passed += delta_time;

        // if(time_passed > 1.0f)
        {
            std::lock_guard guard(statistic_mutex);
            time_passed         = 0.0f;
            reconstruction_time = current_reconstruction_time;
            mapping_time        = current_mapping_time;
            synchronizing_time  = current_synchronizing_time;
            display_time        = delta_time;
        }

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
    }

#ifndef ATCG_HEADLESS
    virtual void onImGuiRender() override
    {
        // return;
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("File"))
        {
            if(ImGui::MenuItem("Save"))
            {
                atcg::Serializer<atcg::ComponentSerializer> serializer(scene);

                serializer.serialize("../Scene/Scene.yaml");
            }

            if(ImGui::MenuItem("Load"))
            {
                scene = atcg::make_ref<atcg::Scene>();
                atcg::Serializer<atcg::ComponentSerializer> serializer(scene);

                serializer.deserialize("../Scene/Scene.yaml");

                hovered_entity = atcg::Entity();
                panel.selectEntity(hovered_entity);
            }

            if(ImGui::MenuItem("Screenshot"))
            {
                auto t  = std::time(nullptr);
                auto tm = *std::localtime(&t);
                std::ostringstream oss;
                oss << "bin/" << "Main" << "_" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << ".png";
                atcg::Renderer::screenshot(scene, camera_controller->getCamera(), 1920, oss.str());
            }

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();


        ImGui::Begin("RIFTCast");

        static bool use_greedy = true;
        if(ImGui::Checkbox("Selection", &use_greedy))
        {
            render_module->setGreedySelection(use_greedy);
        }

        ImGui::End();

        ImGui::Begin("Reconstruction Statistics");

        ImGui::Text(("Reconstruction Time: " + std::to_string(reconstruction_time) + " ms").c_str());
        ImGui::Text(("Mapping Time:   " + std::to_string(mapping_time) + " ms").c_str());
        ImGui::Text(("Synchronizing Time:  " + std::to_string(synchronizing_time) + " ms").c_str());
        ImGui::Separator();
        ImGui::Text(("Render Thread: " + std::to_string(display_time) + " s / " +
                     std::to_string(1.0f / (display_time)) + " fps")
                        .c_str());
        time_collection.addSample(atcg::Renderer::getFrameCounter());
        fps_collection_mean.addSample(1.0f / (display_time + 1e-5f));
        fps_collection.addSample(fps_collection_mean.mean());

        reconstruction_collection_mean.addSample(1.0f / (reconstruction_time / 1000.0f + 1e-5f));
        reconstruction_collection.addSample(reconstruction_collection_mean.mean());

        synchronizing_collection_mean.addSample(1.0f / (synchronizing_time / 1000.0f + 1e-5f));
        synchronizing_collection.addSample(synchronizing_collection_mean.mean());

        mapping_collection_mean.addSample(1.0f / (mapping_time / 1000.0f + 1e-5f));
        mapping_collection.addSample(mapping_collection_mean.mean());

        ImPlot::SetNextAxisLimits(ImAxis_Y1, 0, 100);
        if(ImPlot::BeginPlot("Runtime"))
        {
            ImPlot::SetupAxes("X", "Y", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_None);
            ImPlot::PlotLine("Render Time",
                             time_collection.get(),
                             fps_collection.get(),
                             fps_collection.count(),
                             0,
                             fps_collection.index(),
                             sizeof(float));
            ImPlot::PlotLine("Synchronizing Time",
                             time_collection.get(),
                             synchronizing_collection.get(),
                             synchronizing_collection.count(),
                             0,
                             synchronizing_collection.index(),
                             sizeof(float));
            ImPlot::PlotLine("Reconstruction Time",
                             time_collection.get(),
                             reconstruction_collection.get(),
                             reconstruction_collection.count(),
                             0,
                             reconstruction_collection.index(),
                             sizeof(float));
            ImPlot::PlotLine("Mapping Time",
                             time_collection.get(),
                             mapping_collection.get(),
                             mapping_collection.count(),
                             0,
                             mapping_collection.index(),
                             sizeof(float));
            ImPlot::EndPlot();
        }

        ImGui::End();

        panel.renderPanel();
        hovered_entity = panel.getSelectedEntity();

        atcg::drawGuizmo(scene, hovered_entity, current_operation, camera_controller->getCamera());
    }
#endif

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(RIFTCastLayer::onViewportResized));
#ifndef ATCG_HEADLESS
        dispatcher.dispatch<atcg::MouseMovedEvent>(ATCG_BIND_EVENT_FN(RIFTCastLayer::onMouseMoved));
        dispatcher.dispatch<atcg::MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(RIFTCastLayer::onMousePressed));
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(RIFTCastLayer::onKeyPressed));
        dispatcher.dispatch<atcg::VRButtonPressedEvent>(ATCG_BIND_EVENT_FN(RIFTCastLayer::onVRButtonPressed));
#endif
    }

    bool onViewportResized(atcg::ViewportResizeEvent* event)
    {
        atcg::WindowResizeEvent resize_event(event->getWidth(), event->getHeight());
        camera_controller->onEvent(&resize_event);

        return false;
    }

#ifndef ATCG_HEADLESS
    bool onVRButtonPressed(atcg::VRButtonPressedEvent* event) { return true; }

    bool onKeyPressed(atcg::KeyPressedEvent* event)
    {
        if(event->getKeyCode() == ATCG_KEY_T)
        {
            current_operation = ImGuizmo::OPERATION::TRANSLATE;
        }
        if(event->getKeyCode() == ATCG_KEY_R)
        {
            current_operation = ImGuizmo::OPERATION::ROTATE;
        }
        if(event->getKeyCode() == ATCG_KEY_S)
        {
            current_operation = ImGuizmo::OPERATION::SCALE;
        }

        if(event->getKeyCode() == ATCG_KEY_H)
        {
            hover_mode = !hover_mode;
        }

        if(event->getKeyCode() == ATCG_KEY_K)
        {
            camera_ride = !camera_ride;
        }

        return true;
    }

    bool onMousePressed(atcg::MouseButtonPressedEvent* event)
    {
        if(in_viewport && event->getMouseButton() == ATCG_MOUSE_BUTTON_LEFT && !ImGuizmo::IsOver())
        {
            int id         = atcg::Renderer::getEntityIndex(mouse_pos);
            hovered_entity = id == -1 ? atcg::Entity() : atcg::Entity((entt::entity)id, scene.get());
            panel.selectEntity(hovered_entity);
        }
        return true;
    }

    bool onMouseMoved(atcg::MouseMovedEvent* event)
    {
        const atcg::Application* app = atcg::Application::get();
        glm::ivec2 offset            = app->getViewportPosition();
        int height                   = app->getViewportSize().y;
        mouse_pos                    = glm::vec2(event->getX() - offset.x, height - (event->getY() - offset.y));

        in_viewport =
            mouse_pos.x >= 0 && mouse_pos.y >= 0 && mouse_pos.y < height && mouse_pos.x < app->getViewportSize().x;


        if(hover_mode && in_viewport && !ImGuizmo::IsOver())
        {
            int id         = atcg::Renderer::getEntityIndex(mouse_pos);
            hovered_entity = id == -1 ? atcg::Entity() : atcg::Entity((entt::entity)id, scene.get());
            panel.selectEntity(hovered_entity);
        }

        return false;
    }
#endif

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
    // atcg::ref_ptr<rift::GeometryModule> geometry_module;
    // atcg::ref_ptr<rift::RenderModule> render_module;
    // torch::Tensor cam_valid;
    glm::mat4 model;

    atcg::ref_ptr<atcg::CameraController> camera_controller;

    atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler> panel;

    bool in_viewport = false;

    bool hover_mode  = false;
    bool camera_ride = false;

    glm::vec2 mouse_pos;

    atcg::ref_ptr<atcg::Graph> pointcloud;

    // Visual hull thread
    std::thread visual_hull_thread;
    std::mutex visual_hull_mutex;
    std::atomic_bool running = false;
    torch::Tensor output_vertices, output_faces, output_normals, output_primitives;
    int current_frame = 0;

    // Render thread
    std::thread render_thread;
    std::mutex render_mutex;
    std::atomic_bool start_rendering = false;
    std::atomic_bool rendering_done  = true;
    uint32_t render_input_width, render_input_height;
    glm::mat4 render_input_view, render_input_projection;
    torch::Tensor render_output_img;
    torch::Tensor render_output_depth;
    torch::Tensor render_output_normals;
    glm::mat4 render_output_inv_view_projection;

    // Statistics
    std::mutex statistic_mutex;
    float current_reconstruction_time;
    float current_mapping_time;
    float current_synchronizing_time;

    bool show_render_settings = false;
#ifndef ATCG_HEADLESS
    ImGuizmo::OPERATION current_operation = ImGuizmo::OPERATION::TRANSLATE;
#endif
};

class RIFTCast : public atcg::Application
{
public:
    RIFTCast(const atcg::WindowProps& props) : atcg::Application(props)
    {
        _layer = new RIFTCastLayer("Layer");
        pushLayer(_layer);
    }

    ~RIFTCast() { _layer->destroy(); }

private:
    RIFTCastLayer* _layer;
};

atcg::Application* atcg::createApplication()
{
    atcg::WindowProps props;
    props.vsync = true;
    return new RIFTCast(props);
}