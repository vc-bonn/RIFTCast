#include <riftcast/riftcastkernels.h>

#include <ATen/cuda/ApplyGridUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <DataStructure/Timer.h>

#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

#include <Math/Functions.h>

namespace rift
{
namespace detail
{
ATCG_GLOBAL void visible_primitives_kernel(cudaTextureObject_t texture,
                                           const uint32_t width,
                                           const uint32_t height,
                                           torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> output)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < width * height; tid += num_threads)
    {
        int pixel_x = tid % width;
        int pixel_y = tid / width;

        int pixel_data = tex2D<int>(texture, pixel_x, pixel_y);

        if(pixel_data == -1) continue;

        output[pixel_data] = 1;
    }
}
}    // namespace detail

torch::Tensor computeVisiblePrimitives(const cudaTextureObject_t& texture,
                                       const uint32_t width,
                                       const uint32_t height,
                                       const uint32_t num_primitives)
{
    torch::Tensor result =
        torch::zeros({num_primitives}, torch::TensorOptions {}.dtype(torch::kInt32).device(atcg::GPU));

    auto device = result.device();

    at::cuda::CUDAGuard device_guard {device};
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(width * height, grid, device.index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    detail::visible_primitives_kernel<<<grid, threads, 0, stream>>>(
        texture,
        width,
        height,
        result.packed_accessor32<int, 1, torch::RestrictPtrTraits>());

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    return result;
}

namespace detail
{
ATCG_GLOBAL void
visible_primitives_kernel_batched(const cudaTextureObject_t* textures,
                                  const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> cam_valid,
                                  const uint32_t n_cameras,
                                  const uint32_t width,
                                  const uint32_t height,
                                  torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> output)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < width * height; tid += num_threads)
    {
        if(tid >= width * height) return;

        int pixel_x = tid % width;
        int pixel_y = (tid / width);

        for(int camera_id = 0; camera_id < n_cameras; ++camera_id)
        {
            if(cam_valid[camera_id] == 0) continue;

            int pixel_data = tex2D<int>(textures[camera_id], pixel_x, pixel_y);

            if(pixel_data == -1) continue;

            output[camera_id][pixel_data] = 1;
        }
    }
}
}    // namespace detail

torch::Tensor computeVisiblePrimitivesBatched(const atcg::DeviceBuffer<cudaTextureObject_t>& textures,
                                              const torch::Tensor& cam_valid_mask,
                                              const uint32_t width,
                                              const uint32_t height,
                                              const uint32_t num_cameras,
                                              const uint32_t num_primitives)
{
    torch::Tensor result =
        torch::zeros({num_cameras, num_primitives}, torch::TensorOptions {}.dtype(torch::kInt32).device(atcg::GPU));

    auto device = result.device();

    at::cuda::CUDAGuard device_guard {device};
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(width * height, grid, device.index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    detail::visible_primitives_kernel_batched<<<grid, threads, 0, stream>>>(
        textures.get(),
        cam_valid_mask.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        num_cameras,
        width,
        height,
        result.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    return result;
}
namespace detail
{
ATCG_GLOBAL void
kernelComputeVertexNormals(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> positions,
                           torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> vertex_normals,
                           const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> faces)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < faces.size(0); tid += num_threads)
    {
        int index_0 = faces[tid][0];
        int index_1 = faces[tid][1];
        int index_2 = faces[tid][2];

        glm::vec3 p0 = glm::vec3(positions[index_0][0], positions[index_0][1], positions[index_0][2]);
        glm::vec3 p1 = glm::vec3(positions[index_1][0], positions[index_1][1], positions[index_1][2]);
        glm::vec3 p2 = glm::vec3(positions[index_2][0], positions[index_2][1], positions[index_2][2]);

        glm::vec3 edge_01 = p1 - p0;
        glm::vec3 edge_02 = p2 - p0;
        glm::vec3 normal  = glm::normalize(glm::cross(edge_01, edge_02));

        if(!isfinite(glm::length(normal)))
        {
            continue;
        }

        atomicAdd(&vertex_normals[index_0][0], normal.x);
        atomicAdd(&vertex_normals[index_0][1], normal.y);
        atomicAdd(&vertex_normals[index_0][2], normal.z);
        atomicAdd(&vertex_normals[index_1][0], normal.x);
        atomicAdd(&vertex_normals[index_1][1], normal.y);
        atomicAdd(&vertex_normals[index_1][2], normal.z);
        atomicAdd(&vertex_normals[index_2][0], normal.x);
        atomicAdd(&vertex_normals[index_2][1], normal.y);
        atomicAdd(&vertex_normals[index_2][2], normal.z);
    }
}
}    // namespace detail

void computeVertexNormals(const torch::Tensor& positions, torch::Tensor& vertex_normals, const torch::Tensor& faces)
{
    if(positions.size(0) == 0 || faces.size(0) == 0) return;

    auto device = positions.device();

    at::cuda::CUDAGuard device_guard {device};
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(faces.size(0), grid, device.index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    detail::kernelComputeVertexNormals<<<grid, threads, 0, stream>>>(
        positions.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        vertex_normals.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        faces.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>());

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
}

namespace detail
{
ATCG_GLOBAL void unprojectVertices(torch::PackedTensorAccessor32<uint8_t, 3, torch::RestrictPtrTraits> rgb_image,
                                   const glm::mat4 inv_view_projection,
                                   const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> depth_map,
                                   int* num_vertices,
                                   torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> positions,
                                   torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> colors)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < depth_map.size(0) * depth_map.size(1); tid += num_threads)
    {
        if(tid >= depth_map.size(0) * depth_map.size(1)) return;

        int pixel_x = tid % (int)depth_map.size(1);
        int pixel_y = tid / (int)depth_map.size(1);

        float width  = depth_map.size(1);
        float height = depth_map.size(0);

        glm::vec2 pixel((float)pixel_x / width, (float)pixel_y / height);

        float depth = depth_map[pixel_y][pixel_x][0];

        if(depth == 1.0f) continue;

        int vertex_id = atomicAdd(num_vertices, 1);

        glm::vec4 world_pos_center((pixel.x) * 2.0f - 1.0f, (pixel.y) * 2.0f - 1.0f, 2.0f * depth - 1.0f, 1.0f);

        world_pos_center = inv_view_projection * world_pos_center;
        world_pos_center /= world_pos_center.w;    // 3D Position of patch

        glm::vec3 color = glm::vec3(((float)rgb_image[pixel_y][pixel_x][0]) / 255.0f,
                                    ((float)rgb_image[pixel_y][pixel_x][1]) / 255.0f,
                                    ((float)rgb_image[pixel_y][pixel_x][2]) / 255.0f);

        positions[vertex_id][0] = world_pos_center.x;
        positions[vertex_id][1] = world_pos_center.y;
        positions[vertex_id][2] = world_pos_center.z;

        colors[vertex_id][0] = color.x;
        colors[vertex_id][1] = color.y;
        colors[vertex_id][2] = color.z;
    }
}
}    // namespace detail

std::tuple<torch::Tensor, torch::Tensor>
unprojectVertices(const torch::Tensor& rgb_image, const glm::mat4& inv_view_projection, const torch::Tensor& depth_map)
{
    auto device             = rgb_image.device();
    torch::Tensor positions = torch::empty({rgb_image.numel(), 3}, atcg::TensorOptions::floatDeviceOptions());
    torch::Tensor colors    = torch::empty({rgb_image.numel(), 3}, atcg::TensorOptions::floatDeviceOptions());

    at::cuda::CUDAGuard device_guard {device};
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(rgb_image.size(1) * rgb_image.size(0), grid, device.index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    atcg::dref_ptr<int> dnum_vertices;
    int num_vertices = 0;
    dnum_vertices.upload(&num_vertices);

    detail::unprojectVertices<<<grid, threads, 0, stream>>>(
        rgb_image.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>(),
        inv_view_projection,
        depth_map.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        dnum_vertices.get(),
        positions.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        colors.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    dnum_vertices.download(&num_vertices);

    positions.resize_({num_vertices, 3});
    colors.resize_({num_vertices, 3});

    return std::make_pair(positions, colors);
}

namespace detail
{
ATCG_GLOBAL void unprojectVertices(torch::PackedTensorAccessor32<uint8_t, 3, torch::RestrictPtrTraits> rgb_image,
                                   const glm::mat4 inv_view_projection,
                                   const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> depth_map,
                                   const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> normal_map,
                                   int* num_vertices,
                                   torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> positions,
                                   torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> colors,
                                   torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normals)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < depth_map.size(0) * depth_map.size(1); tid += num_threads)
    {
        if(tid >= depth_map.size(0) * depth_map.size(1)) return;

        int pixel_x = tid % (int)depth_map.size(1);
        int pixel_y = tid / (int)depth_map.size(1);

        float width  = depth_map.size(1);
        float height = depth_map.size(0);

        glm::vec2 pixel((float)pixel_x / width, (float)pixel_y / height);

        float depth = depth_map[pixel_y][pixel_x][0];

        if(depth == 1.0f) continue;

        int vertex_id = atomicAdd(num_vertices, 1);

        glm::vec4 world_pos_center((pixel.x) * 2.0f - 1.0f, (pixel.y) * 2.0f - 1.0f, 2.0f * depth - 1.0f, 1.0f);

        world_pos_center = inv_view_projection * world_pos_center;
        world_pos_center /= world_pos_center.w;    // 3D Position of patch

        glm::vec3 color = glm::vec3(((float)rgb_image[pixel_y][pixel_x][0]) / 255.0f,
                                    ((float)rgb_image[pixel_y][pixel_x][1]) / 255.0f,
                                    ((float)rgb_image[pixel_y][pixel_x][2]) / 255.0f);

        glm::vec3 normal = glm::vec3((normal_map[pixel_y][pixel_x][0]),
                                     (normal_map[pixel_y][pixel_x][1]),
                                     (normal_map[pixel_y][pixel_x][2]));

        positions[vertex_id][0] = world_pos_center.x;
        positions[vertex_id][1] = world_pos_center.y;
        positions[vertex_id][2] = world_pos_center.z;

        colors[vertex_id][0] = color.x;
        colors[vertex_id][1] = color.y;
        colors[vertex_id][2] = color.z;

        normals[vertex_id][0] = normal.x;
        normals[vertex_id][1] = normal.y;
        normals[vertex_id][2] = normal.z;
    }
}
}    // namespace detail

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unprojectVertices(const torch::Tensor& rgb_image,
                                                                          const glm::mat4& inv_view_projection,
                                                                          const torch::Tensor& depth_map,
                                                                          const torch::Tensor& normal_map)
{
    auto device             = rgb_image.device();
    torch::Tensor positions = torch::empty({rgb_image.numel(), 3}, atcg::TensorOptions::floatDeviceOptions());
    torch::Tensor colors    = torch::empty({rgb_image.numel(), 3}, atcg::TensorOptions::floatDeviceOptions());
    torch::Tensor normals   = torch::empty({rgb_image.numel(), 3}, atcg::TensorOptions::floatDeviceOptions());

    at::cuda::CUDAGuard device_guard {device};
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(rgb_image.size(1) * rgb_image.size(0), grid, device.index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    atcg::dref_ptr<int> dnum_vertices;
    int num_vertices = 0;
    dnum_vertices.upload(&num_vertices);

    detail::unprojectVertices<<<grid, threads, 0, stream>>>(
        rgb_image.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>(),
        inv_view_projection,
        depth_map.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        normal_map.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        dnum_vertices.get(),
        positions.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        colors.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        normals.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    dnum_vertices.download(&num_vertices);

    positions.resize_({num_vertices, 3});
    colors.resize_({num_vertices, 3});
    normals.resize_({num_vertices, 3});

    return std::make_tuple(positions, colors, normals);
}

namespace detail
{
ATCG_GLOBAL void
unprojectPointsKernel(const torch::PackedTensorAccessor32<uint8_t, 3, torch::RestrictPtrTraits> rgb_image,
                      const glm::mat4 inv_view_projection,
                      const torch::PackedTensorAccessor32<int16_t, 2, torch::RestrictPtrTraits> depth_map,
                      int* num_vertices,
                      torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> result)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < depth_map.size(0) * depth_map.size(1); tid += num_threads)
    {
        if(tid >= depth_map.size(0) * depth_map.size(1)) return;

        int pixel_x = tid % (int)depth_map.size(1);
        int pixel_y = tid / (int)depth_map.size(1);

        float width  = depth_map.size(1);
        float height = depth_map.size(0);

        glm::vec2 pixel((float)pixel_x / width, (float)pixel_y / height);

        int16_t depth_quantized = depth_map[pixel_y][pixel_x];

        if(depth_quantized < 0) continue;

        float depth_linear = float(depth_quantized) / 5000.0f;
        float depth_ndc    = atcg::Math::linearDepth2ndc(depth_linear, 0.01f, 10.0f);

        int vertex_id = atomicAdd(num_vertices, 1);

        glm::vec4 world_pos_center((pixel.x) * 2.0f - 1.0f, (pixel.y) * 2.0f - 1.0f, depth_ndc, 1.0f);

        world_pos_center = inv_view_projection * world_pos_center;
        world_pos_center /= world_pos_center.w;    // 3D Position of patch

        glm::vec3 color = glm::vec3(((float)rgb_image[pixel_y][pixel_x][0]) / 255.0f,
                                    ((float)rgb_image[pixel_y][pixel_x][1]) / 255.0f,
                                    ((float)rgb_image[pixel_y][pixel_x][2]) / 255.0f);

        result[vertex_id][atcg::VertexSpecification::POSITION_BEGIN + 0] = world_pos_center.x;
        result[vertex_id][atcg::VertexSpecification::POSITION_BEGIN + 1] = world_pos_center.y;
        result[vertex_id][atcg::VertexSpecification::POSITION_BEGIN + 2] = world_pos_center.z;

        result[vertex_id][atcg::VertexSpecification::COLOR_BEGIN + 0] = color.x;
        result[vertex_id][atcg::VertexSpecification::COLOR_BEGIN + 1] = color.y;
        result[vertex_id][atcg::VertexSpecification::COLOR_BEGIN + 2] = color.z;
    }
}
}    // namespace detail

torch::Tensor unprojectVerticesQuantized(const torch::Tensor& rgb_image,
                                         const glm::mat4& inv_view_projection,
                                         const torch::Tensor& depth_map)
{
    auto device = rgb_image.device();
    at::cuda::CUDAGuard device_guard {device};

    torch::Tensor result = torch::empty({rgb_image.size(0) * rgb_image.size(1), atcg::VertexSpecification::VERTEX_SIZE},
                                        atcg::TensorOptions::floatDeviceOptions());

    const auto stream = at::cuda::getCurrentCUDAStream();

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(rgb_image.size(1) * rgb_image.size(0), grid, device.index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    atcg::dref_ptr<int> dnum_vertices;
    int num_vertices = 0;
    dnum_vertices.upload(&num_vertices);

    detail::unprojectPointsKernel<<<grid, threads, 0, stream>>>(
        rgb_image.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>(),
        inv_view_projection,
        depth_map.packed_accessor32<int16_t, 2, torch::RestrictPtrTraits>(),
        dnum_vertices.get(),
        result.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    dnum_vertices.download(&num_vertices);

    result.resize_({num_vertices, atcg::VertexSpecification::VERTEX_SIZE});

    return result;
}

namespace detail
{
ATCG_GLOBAL void quantize_kernel(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> depth_data,
                                 const float n,
                                 const float f,
                                 torch::PackedTensorAccessor32<uint16_t, 2, torch::RestrictPtrTraits> depth_quantized)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < depth_data.size(0) * depth_data.size(1); tid += num_threads)
    {
        if(tid >= depth_data.size(0) * depth_data.size(1)) return;

        int pixel_x = tid % (int)depth_data.size(1);
        int pixel_y = tid / (int)depth_data.size(1);

        float depth = depth_data[pixel_y][pixel_x][0];

        depth = atcg::Math::ndc2linearDepth(atcg::Math::uv2ndc(depth), n, f);

        depth_quantized[pixel_y][pixel_x] = (uint16_t)(depth * 5000.0f);
    }
}
}    // namespace detail

torch::Tensor quantize(const torch::Tensor& depth_data, const float n, const float f)
{
    auto device = depth_data.device();
    at::cuda::CUDAGuard device_guard {device};

    torch::Tensor result = torch::empty({depth_data.size(0), depth_data.size(1)},
                                        torch::TensorOptions {}.dtype(torch::kUInt16).device(atcg::GPU));

    const auto stream = at::cuda::getCurrentCUDAStream();

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(depth_data.size(1) * depth_data.size(0), grid, device.index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    detail::quantize_kernel<<<grid, threads, 0, stream>>>(
        depth_data.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        n,
        f,
        result.packed_accessor32<uint16_t, 2, torch::RestrictPtrTraits>());

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));


    return result;
}
}    // namespace rift