#include <riftcast/RenderModule.h>

#include "../shader/num_cameras.glsl"

#include <Renderer/ContextManager.h>
#include <Renderer/Renderer.h>
#include <riftcast/riftcastkernels.h>

#include <c10/cuda/CUDAStream.h>
#include <torch/cuda.h>

namespace rift
{
class RenderModule::Impl
{
public:
    Impl();
    ~Impl();

    void init(const uint32_t device_idx,
              const atcg::ref_ptr<rift::DatasetImporter>& dataloader,
              const atcg::JPEGBackend& backend);
    torch::Tensor findBestCameras(torch::Tensor& target,
                                  const torch::Tensor& camera_visibilities,
                                  const torch::Tensor& cam_valid_last_frame);

    torch::Tensor findBestCamerasViaDirection(const atcg::ref_ptr<atcg::PerspectiveCamera>& camera);

    uint32_t device_idx;
    atcg::ref_ptr<rift::DatasetImporter> dataloader;

    atcg::ref_ptr<atcg::Context> render_context             = nullptr;
    atcg::ref_ptr<atcg::ShaderManagerSystem> shader_manager = nullptr;
    atcg::ref_ptr<atcg::RendererSystem> renderer            = nullptr;
    atcg::ref_ptr<atcg::Graph> visual_hull                  = nullptr;
    atcg::ref_ptr<atcg::Texture3D> rgb_textures             = nullptr;
    atcg::ref_ptr<atcg::Framebuffer> primitive_buffer       = nullptr;
    atcg::ref_ptr<atcg::Framebuffer> framebuffer            = nullptr;
    atcg::ref_ptr<atcg::JPEGDecoder> decoder;
    std::array<atcg::ref_ptr<atcg::Framebuffer>, NUM_CAMERAS> depth_maps;
    torch::Tensor rgb_valid;

    glm::vec4 clear_color     = glm::vec4(1);
    bool use_greedy_selection = true;
};

RenderModule::Impl::Impl() {}

RenderModule::Impl::~Impl()
{
    render_context->makeCurrent();

    // Manually free opengl buffer before context is destroyed
    visual_hull.reset();
    framebuffer.reset();
    primitive_buffer.reset();
    rgb_textures.reset();
    depth_maps = {};
    renderer.reset();
    shader_manager.reset();

    atcg::ContextManager::destroyContext(render_context);
}

void RenderModule::Impl::init(const uint32_t device_idx,
                              const atcg::ref_ptr<rift::DatasetImporter>& dataloader,
                              const atcg::JPEGBackend& backend)
{
    this->device_idx = device_idx;
    this->dataloader = dataloader;

    render_context = atcg::ContextManager::createContext(device_idx);

    shader_manager = atcg::make_ref<atcg::ShaderManagerSystem>();

    renderer = atcg::make_ref<atcg::RendererSystem>();
    renderer->init(1600, 900, render_context, shader_manager);
    renderer->toggleCulling(true);
    renderer->toggleMSAA(false);

    atcg::ref_ptr<atcg::Shader> depth_shader = atcg::make_ref<atcg::Shader>("RIFTCast/src/riftcast/_C/shader/"
                                                                            "depth_pass.vs",
                                                                            "RIFTCast/src/riftcast/_C/shader/"
                                                                            "depth_pass.fs");
    shader_manager->addShader("depth_pass_vci", depth_shader);
    atcg::ref_ptr<atcg::Shader> vci_shader = atcg::make_ref<atcg::Shader>("RIFTCast/src/riftcast/_C/shader/rift.vs",
                                                                          "RIFTCast/src/riftcast/_C/shader/"
                                                                          "rift.fs");
    shader_manager->addShader("rift", vci_shader);
    atcg::ref_ptr<atcg::Shader> primitive_shader = atcg::make_ref<atcg::Shader>("RIFTCast/src/riftcast/_C/shader/"
                                                                                "primitive_pass.vs",
                                                                                "RIFTCast/src/riftcast/_C/shader/"
                                                                                "primitive_pass.fs");
    shader_manager->addShader("primitive", primitive_shader);

    visual_hull = atcg::Graph::createTriangleMesh();

    atcg::TextureSpecification spec;
    spec.width               = dataloader->width();
    spec.height              = dataloader->height();
    spec.depth               = NUM_CAMERAS;
    spec.sampler.filter_mode = atcg::TextureFilterMode::NEAREST;
    rgb_textures             = atcg::Texture3D::create(spec);

    for(int i = 0; i < NUM_CAMERAS; ++i)
    {
        // Depth Maps
        atcg::ref_ptr<atcg::Framebuffer> camera_buffer =
            atcg::make_ref<atcg::Framebuffer>(dataloader->width(), dataloader->height());
        atcg::TextureSpecification spec;
        spec.width                               = camera_buffer->width();
        spec.height                              = camera_buffer->height();
        spec.format                              = atcg::TextureFormat::RFLOAT;
        spec.sampler.filter_mode                 = atcg::TextureFilterMode::NEAREST;
        atcg::ref_ptr<atcg::Texture2D> depth_map = atcg::Texture2D::create(spec);
        camera_buffer->attachTexture(depth_map);
        camera_buffer->attachDepth();
        camera_buffer->complete();

        depth_maps[i] = camera_buffer;
    }

    rgb_valid = torch::zeros({dataloader->num_cameras()}, atcg::TensorOptions::int32DeviceOptions());
    rgb_valid.index_put_({torch::indexing::Slice(0, NUM_CAMERAS)}, 1);

    decoder = atcg::make_ref<atcg::JPEGDecoder>(NUM_CAMERAS,
                                                dataloader->width(),
                                                dataloader->height(),
                                                dataloader->flip_images(),
                                                backend);

    render_context->deactivate();
}

torch::Tensor RenderModule::Impl::findBestCameras(torch::Tensor& target,
                                                  const torch::Tensor& camera_visibilities,
                                                  const torch::Tensor& cam_valid_last_frame)
{
    int n_cameras           = camera_visibilities.size(0);
    torch::Tensor cam_valid = torch::zeros({n_cameras}, atcg::TensorOptions::int32DeviceOptions());

    // Compute last frame's camera coverage
    float coverage = 0.0f;
    if(cam_valid_last_frame.numel() > 0)
    {
        torch::Tensor camera_visibilities_last_frame = camera_visibilities.index({cam_valid_last_frame == 1});
        torch::Tensor visible_both_last_frame        = camera_visibilities_last_frame * target;

        visible_both_last_frame = torch::clamp(torch::sum(visible_both_last_frame, {0}), 0, 1);

        coverage = visible_both_last_frame.sum().item<float>() / target.sum().item<float>();
    }

    if(coverage > 0.85f)
    {
        return cam_valid_last_frame.clone();
    }


    for(int i = 0; i < NUM_CAMERAS; ++i)
    {
        torch::Tensor visible_both = target * camera_visibilities;
        int best_camera            = torch::argmax(torch::sum(visible_both, 1)).item<int>();
        target                     = target * (1.0f - visible_both[best_camera]);
        cam_valid.index_put_({best_camera}, 1);
    }

    if(cam_valid.sum().item<int>() < NUM_CAMERAS)
    {
        cam_valid.zero_();
        cam_valid.index_put_({torch::indexing::Slice(0, NUM_CAMERAS)}, 1);
    }

    return cam_valid;
}

torch::Tensor RenderModule::Impl::findBestCamerasViaDirection(const atcg::ref_ptr<atcg::PerspectiveCamera>& camera)
{
    const auto& cameras     = dataloader->getCameras();
    torch::Tensor cam_valid = torch::zeros({(int)cameras.size()}, atcg::TensorOptions::int32DeviceOptions());

    torch::Tensor scores = torch::zeros({(int)cameras.size()}, atcg::TensorOptions::floatHostOptions());
    for(int i = 0; i < cameras.size(); ++i)
    {
        scores[i] = glm::dot(cameras[i].cam->getDirection(), camera->getDirection());
    }

    auto best_indices = torch::argsort(scores, -1, true);

    cam_valid[best_indices[0]] = 1;
    cam_valid[best_indices[1]] = 1;
    cam_valid[best_indices[2]] = 1;

    return cam_valid;
}

RenderModule::RenderModule()
{
    impl = std::make_unique<Impl>();
}

RenderModule::~RenderModule() {}

void RenderModule::init(const uint32_t device_idx,
                        const atcg::ref_ptr<rift::DatasetImporter>& dataloader,
                        const atcg::JPEGBackend& backend)
{
    impl->init(device_idx, dataloader, backend);
}

void RenderModule::updateState(const GeometryReconstruction& reconstruction,
                               const atcg::ref_ptr<atcg::PerspectiveCamera>& camera,
                               const uint32_t width,
                               const uint32_t height)
{
    at::cuda::CUDAStream torch_stream = c10::cuda::getCurrentCUDAStream();
    impl->renderer->use();
    if(reconstruction.vertices.numel() > 0 && reconstruction.faces.numel() > 0)
    {
        impl->visual_hull->resizeVertices(reconstruction.vertices.size(0));
        impl->visual_hull->resizeFaces(reconstruction.faces.size(0));
        impl->visual_hull->getDevicePositions().index_put_({torch::indexing::Slice(), torch::indexing::Slice()},
                                                           reconstruction.vertices);
        impl->visual_hull->getDeviceFaces().index_put_({torch::indexing::Slice(), torch::indexing::Slice()},
                                                       reconstruction.faces);
        impl->visual_hull->getDeviceNormals().index_put_({torch::indexing::Slice(), torch::indexing::Slice()},
                                                         reconstruction.normals);

        // Mapper is not used by vh thread right now

        if(!impl->primitive_buffer || impl->primitive_buffer->width() != width ||
           impl->primitive_buffer->height() != height)
        {
            impl->primitive_buffer = atcg::make_ref<atcg::Framebuffer>(width, height);
            atcg::TextureSpecification spec_int;
            spec_int.width               = width;
            spec_int.height              = height;
            spec_int.format              = atcg::TextureFormat::RINT;
            spec_int.sampler.filter_mode = atcg::TextureFilterMode::NEAREST;
            impl->primitive_buffer->attachTexture(atcg::Texture2D::create(spec_int));
            impl->primitive_buffer->attachDepth();
            impl->primitive_buffer->complete();
        }

        impl->primitive_buffer->use();
        impl->renderer->clear();
        impl->renderer->setViewport(0, 0, width, height);
        int value = -1;
        impl->primitive_buffer->getColorAttachement(0)->fill(&value);

        torch_stream.synchronize();

        impl->renderer->draw(impl->visual_hull,
                             camera,
                             glm::mat4(1),
                             glm::vec3(1),
                             impl->shader_manager->getShader("primitive"));

        impl->renderer->finish();

        atcg::textureObject handle =
            impl->primitive_buffer->getColorAttachement(0)->getTextureObject(0, glm::vec4(1), false, false);

        auto currently_visible = rift::computeVisiblePrimitives(handle, width, height, impl->visual_hull->n_faces());


        if(impl->use_greedy_selection)
        {
            impl->rgb_valid =
                impl->findBestCameras(currently_visible, reconstruction.visible_primitives, impl->rgb_valid);
        }
        else
        {
            impl->rgb_valid = impl->findBestCamerasViaDirection(camera);
        }
        torch_stream.synchronize();
        impl->primitive_buffer->getColorAttachement(0)->unmapDevicePointers();
    }

    // 4. Load images
    auto rgb_images = impl->dataloader->getImages(reconstruction.current_frame, impl->rgb_valid);
    auto output     = impl->decoder->decompressImages(rgb_images);

    auto texture = impl->rgb_textures->getTextureArray();
    impl->decoder->copyToOutput(texture);

    // 5. Render visual hull
    if(!impl->framebuffer || impl->framebuffer->width() != width || impl->framebuffer->height() != height)
    {
        impl->framebuffer = atcg::make_ref<atcg::Framebuffer>(width, height);
        impl->framebuffer->attachColor();
        impl->framebuffer->attachDepth();
        atcg::TextureSpecification spec_int;
        spec_int.width  = width;
        spec_int.height = height;
        spec_int.format = atcg::TextureFormat::RINT;
        impl->framebuffer->attachTexture(atcg::Texture2D::create(spec_int));    // Entity ids
        impl->framebuffer->attachTexture(atcg::Texture2D::create(spec_int));    // visibility
        atcg::TextureSpecification spec_float;
        spec_float.width  = width;
        spec_float.height = height;
        spec_float.format = atcg::TextureFormat::RFLOAT;
        impl->framebuffer->attachTexture(atcg::Texture2D::create(spec_float));    // depth
        atcg::TextureSpecification spec_float3;
        spec_float3.width  = width;
        spec_float3.height = height;
        spec_float3.format = atcg::TextureFormat::RGBAFLOAT;
        impl->framebuffer->attachTexture(atcg::Texture2D::create(spec_float3));    // normals
        impl->framebuffer->complete();
    }

    const std::vector<rift::CameraData>& cameras = impl->dataloader->getCameras();
    torch::Tensor host_flags                     = impl->rgb_valid.to(torch::Device(torch::kCPU));
    auto clear_color                             = impl->renderer->getClearColor();
    impl->renderer->setClearColor(glm::vec4(1));
    impl->renderer->setCullFace(atcg::CullMode::ATCG_FRONT_FACE_CULLING);
    int current_id = 0;
    for(uint32_t id = 0; id < impl->dataloader->num_cameras(); ++id)
    {
        auto vci_camera = cameras[id].cam;

        if(host_flags.index({(int)id}).item<int>() == 0) continue;

        // Render depth map
        auto cam_buffer = impl->depth_maps[current_id];
        cam_buffer->use();

        impl->renderer->setViewport(0, 0, cam_buffer->width(), cam_buffer->height());
        impl->renderer->clear();
        auto& shader = impl->shader_manager->getShader("depth_pass_vci");
        impl->renderer->draw(impl->visual_hull, vci_camera, glm::mat4(1), glm::vec3(1), shader);
        ++current_id;
    }
    impl->renderer->setCullFace(atcg::CullMode::ATCG_BACK_FACE_CULLING);
    // impl->renderer->toggleCulling(false);
    impl->renderer->setClearColor(clear_color);

    torch_stream.synchronize();
    impl->rgb_textures->unmapDevicePointers();

    impl->renderer->finish();
}

atcg::ref_ptr<atcg::Framebuffer> RenderModule::renderFrame(const atcg::ref_ptr<atcg::PerspectiveCamera>& camera)
{
    at::cuda::CUDAStream torch_stream = c10::cuda::getCurrentCUDAStream();
    impl->renderer->use();

    impl->framebuffer->use();
    impl->renderer->setViewport(0, 0, impl->framebuffer->width(), impl->framebuffer->height());
    impl->renderer->setClearColor(impl->clear_color);
    impl->renderer->clear();
    float dummy = 1.0f;
    int mone    = -1;
    impl->framebuffer->getColorAttachement(2)->fill(&mone);
    impl->framebuffer->getColorAttachement(3)->fill(&dummy);
    auto shader = impl->shader_manager->getShader("rift");

    uint32_t texture_tensor_id = impl->renderer->popTextureID();
    impl->rgb_textures->use(texture_tensor_id);
    shader->setInt("texture_tensor", texture_tensor_id);

    uint32_t ids[NUM_CAMERAS];

    auto chosen_cams = torch::arange((int)impl->dataloader->num_cameras(), atcg::TensorOptions::int32HostOptions())
                           .index({impl->rgb_valid.to(atcg::CPU) == 1});

    const std::vector<rift::CameraData>& cameras = impl->dataloader->getCameras();
    for(int i = 0; i < NUM_CAMERAS; ++i)
    {
        ids[i] = impl->renderer->popTextureID();
        shader->setInt("depth_maps[" + std::to_string(i) + "]", ids[i]);
        impl->depth_maps[i]->getColorAttachement(0)->use(ids[i]);
        shader->setMat4("cam_view_projections[" + std::to_string(i) + "]",
                        cameras[chosen_cams[i].item<int>()].cam->getViewProjection());
        shader->setVec3("cam_positions[" + std::to_string(i) + "]",
                        cameras[chosen_cams[i].item<int>()].cam->getPosition());
    }

    shader->setVec3("cam_position_mapping", camera->getPosition());
    shader->setMat4("cam_view_projection_mapping", camera->getViewProjection());

    impl->renderer->draw(impl->visual_hull, camera, glm::mat4(1), glm::vec3(1.0f), shader);

    for(int i = 0; i < NUM_CAMERAS; ++i)
    {
        impl->renderer->pushTextureID(ids[i]);
    }

    impl->renderer->pushTextureID(texture_tensor_id);

    impl->renderer->finish();

    return impl->framebuffer;
}

torch::Tensor RenderModule::getChosenCameraIndices() const
{
    return impl->rgb_valid;
}

atcg::ref_ptr<atcg::ShaderManagerSystem> RenderModule::getShaderManager() const
{
    return impl->shader_manager;
}

void RenderModule::setBackgroundColor(const glm::vec4& color)
{
    impl->clear_color = color;
}

void RenderModule::setGreedySelection(const bool use_greedy_selection)
{
    impl->use_greedy_selection = use_greedy_selection;
}
}    // namespace rift