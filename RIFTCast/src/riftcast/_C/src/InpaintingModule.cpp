#include <riftcast/InpaintingModule.h>

#ifndef _WIN32
    #include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#include <torch/torch.h>
#include <DataStructure/TorchUtils.h>

namespace rift
{
class InpaintingModule::Impl
{
public:
    Impl();
    ~Impl();

    void init(const std::string& base_path);

#ifndef _WIN32
    atcg::ref_ptr<torch::inductor::AOTIModelContainerRunnerCuda> runner_single;
    atcg::ref_ptr<torch::inductor::AOTIModelContainerRunnerCuda> runner_memory;
#endif

    torch::Tensor memory;
    uint32_t frame_idx = 0;
};

InpaintingModule::Impl::Impl() {}

InpaintingModule::Impl::~Impl() {}

InpaintingModule::InpaintingModule()
{
    impl = std::make_unique<Impl>();
}

void InpaintingModule::Impl::init(const std::string& base_path)
{
#ifndef _WIN32
    runner_single = atcg::make_ref<torch::inductor::AOTIModelContainerRunnerCuda>(base_path + "/single/"
                                                                                              "model_single.so");
    runner_memory = atcg::make_ref<torch::inductor::AOTIModelContainerRunnerCuda>(base_path + "/memory/"
                                                                                              "model_memory.so");
#endif
    memory = torch::tensor({}, atcg::TensorOptions::floatDeviceOptions());
}

InpaintingModule::~InpaintingModule() {}

InpaintingModule::InpaintingModule(const std::string& base_path) : InpaintingModule()
{
    impl->init(base_path);
}

torch::Tensor InpaintingModule::inpaint(const torch::Tensor& image, const torch::Tensor& mask)
{
#ifndef _WIN32
    torch::Tensor img_tensor  = image;
    torch::Tensor mask_tensor = mask;

    int original_width  = img_tensor.size(1);
    int original_height = img_tensor.size(0);

    img_tensor  = img_tensor.permute({2, 0, 1}).unsqueeze(0).to(torch::kFloat32) / 255.;
    img_tensor  = img_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    mask_tensor = mask_tensor.permute({2, 0, 1}).unsqueeze(0);

    torch::Tensor resized_img = torch::nn::functional::interpolate(
        img_tensor,
        torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t> {240, 432}).mode(torch::kArea));

    torch::Tensor resized_mask = torch::nn::functional::interpolate(
        mask_tensor,
        torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t> {240, 432}).mode(torch::kArea));
    resized_mask = torch::clamp(resized_mask, 0, 1);

    resized_img = resized_img * 2 - 1;

    auto masked_img = (resized_img * (1.0f - resized_mask)).unsqueeze(0);

    std::vector<torch::Tensor> inputs = {masked_img, impl->memory};

    torch::autograd::variable_list outputs;
    if(impl->frame_idx < 6)
    {
        outputs      = impl->runner_single->run(inputs);
        impl->memory = torch::cat({impl->memory, outputs[1].unsqueeze(0)});
    }
    else
    {
        outputs      = impl->runner_memory->run(inputs);
        impl->memory = torch::roll(impl->memory, {-1});
        impl->memory.index_put_({-1}, outputs[1].unsqueeze(0));
    }
    ++impl->frame_idx;

    auto pred_img = outputs[0];

    pred_img = ((pred_img + 1.0f) / 2.0f);

    pred_img = torch::nn::functional::interpolate(pred_img,
                                                  torch::nn::functional::InterpolateFuncOptions()
                                                      .size(std::vector<int64_t> {original_height, original_width})
                                                      .mode(torch::kArea));

    // auto masked_areas = (mask_tensor == 1.0f);
    // img_tensor.index_put_({masked_areas}, pred_img.index({masked_areas}));
    pred_img = img_tensor * (1.0f - mask_tensor) + pred_img * mask_tensor;
    pred_img = (pred_img * 255).squeeze(0).to(torch::kUInt8).permute({1, 2, 0});    //.reshape({432, 240, 3});

    return pred_img;
#else
    return image;
#endif
}


}    // namespace rift