#include <riftcast/GeometryModule.h>

#include <Renderer/Context.h>
#include <Renderer/ContextManager.h>
#include <Renderer/Renderer.h>
#include <Renderer/ShaderManager.h>
#include <riftcast/riftcastkernels.h>

#include <maskcompression/decompress.h>
#include <torchhull/visual_hull.h>
#include <torchhull/gaussian_blur.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace rift
{
class GeometryModule::Impl
{
public:
    Impl();
    ~Impl();

    void init(const uint32_t device_idx, const atcg::ref_ptr<rift::DatasetImporter>& dataloader);
    void renderPrimitiveMaps(const atcg::ref_ptr<atcg::Graph>& mesh, const torch::Tensor& cam_valid_mask);
    void mapPrimitiveMaps(const torch::Tensor& cam_valid);
    void unmapPrimitiveMaps();

    atcg::ref_ptr<atcg::Context> visual_hull_context;
    atcg::ref_ptr<atcg::RendererSystem> visual_hull_renderer;
    atcg::ref_ptr<atcg::ShaderManagerSystem> visual_hull_shader_manager;
    atcg::ref_ptr<atcg::Graph> vh_graph;
    torch::Tensor view_projection_tensor;

    atcg::ref_ptr<rift::DatasetImporter> dataloader;
    uint32_t device_idx;

    // Camera data
    std::vector<rift::CameraData> cameras;
    uint32_t primitive_width = 400;
    atcg::ref_ptr<atcg::Shader> primitive_shader;
    std::vector<atcg::ref_ptr<atcg::Framebuffer>> camera_primitive_buffers;
    std::vector<cudaTextureObject_t> host_primitive_texture_handles;
    atcg::DeviceBuffer<cudaTextureObject_t> primitive_texture_handles;
    bool primitive_mapped_vci = false;
};

GeometryModule::Impl::Impl() {}

GeometryModule::Impl::~Impl()
{
    if(visual_hull_renderer) visual_hull_renderer->use();
    // Manual clean up because the context needs to be destroyed last
    if(vh_graph) vh_graph.reset();
    if(primitive_shader) primitive_shader.reset();
    camera_primitive_buffers.clear();
    if(visual_hull_renderer) visual_hull_renderer.reset();
    if(visual_hull_shader_manager) visual_hull_shader_manager.reset();
    if(visual_hull_context) atcg::ContextManager::destroyContext(visual_hull_context);
}

void GeometryModule::Impl::init(const uint32_t device_idx, const atcg::ref_ptr<rift::DatasetImporter>& dataloader)
{
    this->device_idx = device_idx;
    this->dataloader = dataloader;

    torch::Device visual_hull_device(torch::kCUDA, device_idx);
    SET_DEVICE(device_idx);

    at::cuda::CUDAGuard device_guard(device_idx);
    visual_hull_context = atcg::ContextManager::createContext(device_idx);

    visual_hull_renderer       = atcg::make_ref<atcg::RendererSystem>();
    visual_hull_shader_manager = atcg::make_ref<atcg::ShaderManagerSystem>();
    visual_hull_renderer->init(dataloader->width(),
                               dataloader->height(),
                               visual_hull_context,
                               visual_hull_shader_manager);
    visual_hull_renderer->toggleCulling(false);
    visual_hull_renderer->toggleMSAA(false);

    vh_graph = atcg::Graph::createTriangleMesh();

    view_projection_tensor = dataloader->getViewProjectionTensor().to(visual_hull_device);

    cameras = dataloader->getCameras();
    camera_primitive_buffers.resize(cameras.size());
    host_primitive_texture_handles.resize(cameras.size());
    primitive_texture_handles.create(cameras.size());
    for(uint32_t i = 0; i < cameras.size(); ++i)
    {
        atcg::ref_ptr<atcg::PerspectiveCamera> cam = cameras[i].cam;

        // Primitive Maps
        atcg::ref_ptr<atcg::Framebuffer> primitive_buffers =
            atcg::make_ref<atcg::Framebuffer>(primitive_width, (int)(primitive_width / cam->getAspectRatio()));
        atcg::TextureSpecification int_spec;
        int_spec.width                               = primitive_width;
        int_spec.height                              = (int)(primitive_width / cam->getAspectRatio());
        int_spec.format                              = atcg::TextureFormat::RINT;
        atcg::ref_ptr<atcg::Texture2D> primitive_map = atcg::Texture2D::create(int_spec);
        primitive_buffers->attachTexture(primitive_map);
        primitive_buffers->attachDepth();
        primitive_buffers->complete();

        camera_primitive_buffers[i] = primitive_buffers;
    }

    primitive_shader = atcg::make_ref<atcg::Shader>("RIFTCast/src/riftcast/_C/shader/primitive_pass.vs",
                                                    "RIFTCast/src/riftcast/_C/shader/primitive_pass.fs");

    visual_hull_context->deactivate();
}

void GeometryModule::Impl::renderPrimitiveMaps(const atcg::ref_ptr<atcg::Graph>& mesh,
                                               const torch::Tensor& cam_valid_mask)
{
    if(mesh->n_vertices() == 0 || mesh->n_faces() == 0)
    {
        return;
    }

    // Compute primitive map for each camera
    torch::Tensor host_flags = cam_valid_mask.to(torch::Device(torch::kCPU));
    for(uint32_t id = 0; id < cameras.size(); ++id)
    {
        if(host_flags.index({(int)id}).item<int>() == 0) continue;
        auto vci_camera = cameras[id].cam;

        // Render depth map
        auto primitive_buffer = camera_primitive_buffers[id];
        primitive_buffer->use();

        visual_hull_renderer->setViewport(0, 0, primitive_buffer->width(), primitive_buffer->height());
        visual_hull_renderer->clear();
        int value = -1;
        primitive_buffer->getColorAttachement(0)->fill(&value);
        visual_hull_renderer->draw(mesh, vci_camera, glm::mat4(1), glm::vec3(1), primitive_shader);
    }

    visual_hull_renderer->finish();
}

void GeometryModule::Impl::mapPrimitiveMaps(const torch::Tensor& cam_valid)
{
    if(primitive_mapped_vci) return;

    auto host_flags = cam_valid.to(torch::kCPU);
    for(uint32_t id = 0; id < cameras.size(); ++id)
    {
        // if(host_flags.index({(int)id}).item<int>() == 0) continue;

        auto primitive_buffer    = camera_primitive_buffers[id];
        atcg::textureArray array = primitive_buffer->getColorAttachement(0)->getTextureArray();

        cudaResourceDesc resDesc = {};
        resDesc.resType          = cudaResourceTypeArray;
        resDesc.res.array.array  = array;

        cudaTextureDesc texDesc  = {};
        texDesc.addressMode[0]   = cudaAddressModeBorder;
        texDesc.addressMode[1]   = cudaAddressModeBorder;
        texDesc.filterMode       = cudaFilterModePoint;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaTextureObject_t texObj = 0;
        CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        host_primitive_texture_handles[id] = texObj;
    }

    primitive_texture_handles.upload(host_primitive_texture_handles.data());

    primitive_mapped_vci = true;
}

void GeometryModule::Impl::unmapPrimitiveMaps()
{
    if(!primitive_mapped_vci) return;

    for(int i = 0; i < cameras.size(); ++i)
    {
        CUDA_SAFE_CALL(cudaDestroyTextureObject(host_primitive_texture_handles[i]));
        camera_primitive_buffers[i]->getColorAttachement()->unmapDevicePointers();
    }

    primitive_mapped_vci = false;
}

GeometryModule::GeometryModule()
{
    impl = std::make_unique<Impl>();
}

GeometryModule::~GeometryModule() {}

void GeometryModule::init(const uint32_t device_idx, const atcg::ref_ptr<rift::DatasetImporter>& dataloader)
{
    impl->init(device_idx, dataloader);
}

GeometryReconstruction
GeometryModule::compute_geometry(const glm::mat4& model, const torch::Tensor& cam_valid, const uint32_t frame)
{
    GeometryReconstruction reconstruction;

    at::cuda::CUDAGuard device_guard(impl->device_idx);

    auto torch_stream = at::cuda::getCurrentCUDAStream();

    impl->visual_hull_renderer->use();

    // Load masks
    torch::Tensor masks;

    // Decompress masks
    auto masks_compressed =
        frame == -1 ? impl->dataloader->getMasks(cam_valid) : impl->dataloader->getMasks(frame, cam_valid);
    std::array<int, 2> data = {impl->dataloader->height(), impl->dataloader->width()};
    masks                   = maskcompression::decompress(masks_compressed, data, impl->dataloader->flip_masks());
    masks.unsqueeze_(-1);

    // Visual Hull
    glm::vec3 lower_corner = model * glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f);
    float scale_x          = glm::length(glm::vec3(model[0]));
    float scale_y          = glm::length(glm::vec3(model[1]));
    float scale_z          = glm::length(glm::vec3(model[2]));
    float scale            = glm::max(scale_x, glm::max(scale_y, scale_z));

    torch::Tensor valid = cam_valid.to(torch::kBool);
    // input_{vertices|faces} tensor are created by the main thread to to OpenGL
    if(impl->dataloader->enable_smoothing())
    {
        masks = torchhull::gaussian_blur(masks,
                                         impl->dataloader->kernel_size(),
                                         impl->dataloader->sigma(),
                                         true,
                                         torch::kFloat32);
    }
    auto [vertices, faces] = torchhull::visual_hull(masks,    //.index({valid}),
                                                    impl->view_projection_tensor.index({valid}),
                                                    impl->dataloader->level(),
                                                    {lower_corner.x, lower_corner.y, lower_corner.z},
                                                    scale * 2.0f,
                                                    /*masks_partial=*/impl->dataloader->partial_masks(),
                                                    /*transform_convention=*/"opengl",
                                                    /*unique=*/true);

    torch::Tensor normals = torch::zeros_like(vertices);

    torch::Tensor visible_primitives_per_camera;
    if(vertices.size(0) > 0)
    {
        impl->vh_graph->resizeVertices(vertices.size(0));
        impl->vh_graph->resizeFaces(faces.size(0));
        impl->vh_graph->getDevicePositions().index_put_({torch::indexing::Slice(), torch::indexing::Slice()}, vertices);
        impl->vh_graph->getDeviceFaces().index_put_({torch::indexing::Slice(), torch::indexing::Slice()}, faces);

        torch_stream.synchronize();
        impl->vh_graph->unmapAllDevicePointers();

        impl->renderPrimitiveMaps(impl->vh_graph, cam_valid);
        impl->mapPrimitiveMaps(cam_valid);
        visible_primitives_per_camera = computeVisiblePrimitivesBatched(impl->primitive_texture_handles,
                                                                        cam_valid,
                                                                        impl->camera_primitive_buffers[0]->width(),
                                                                        impl->camera_primitive_buffers[0]->height(),
                                                                        impl->cameras.size(),
                                                                        faces.size(0));
        torch_stream.synchronize();
        impl->unmapPrimitiveMaps();
    }

    rift::computeVertexNormals(vertices, normals, faces);

    reconstruction.vertices           = vertices;
    reconstruction.faces              = faces;
    reconstruction.normals            = normals;
    reconstruction.visible_primitives = visible_primitives_per_camera;
    reconstruction.current_frame      = impl->dataloader->getLastAvailableFrame();

    return reconstruction;
}

std::vector<atcg::ref_ptr<atcg::Framebuffer>> GeometryModule::getPrimitiveBuffers() const
{
    return impl->camera_primitive_buffers;
}
}    // namespace rift