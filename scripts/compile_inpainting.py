import git
import os
import sys
import gdown
import torch
import importlib

sys.path.insert(0, "inpainting")


def clone_dstt():
    repo_url = "https://github.com/guillaume-thiry/towards-online-video-inpainting"

    clone_dir = "inpainting"

    if not os.path.exists(clone_dir):
        print(f"Cloning repository from {repo_url} to {clone_dir}...")
        git.Repo.clone_from(repo_url, clone_dir)
        print("Repository cloned successfully.")
    else:
        print(f"Directory '{clone_dir}' already exists. Skipping clone.")


def download_weights():
    checkpoint_dir = os.path.join("inpainting", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    file_id = "1XheHlarQT4Yh5Myxvi6hB6cxOLIbDqNL"
    output_path = os.path.join(checkpoint_dir, "DSTT.pth")  # Change name/ext as needed
    gdown.download(id=file_id, output=output_path, quiet=False)

    print(f"File downloaded to: {output_path}")


def compile_models():
    torch.set_float32_matmul_precision("high")
    device = "cuda:0"
    # Script the entire model class
    model_name = "DSTT_OM"
    # Loading
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module("inpainting.model." + model_name)
    model = net.InpaintGenerator().to(device)
    data = torch.load("inpainting/checkpoints/DSTT.pth", map_location=device)
    model.load_state_dict(data["netG"])
    model.eval()

    torch.random.manual_seed(1337)
    input_img = torch.randn((1, 1, 3, 240, 432), device=device, dtype=torch.float32)
    memory = torch.Tensor().to(
        device
    )  # torch.randn([6, 8, 1, 720, 512], device=device, dtype=torch.float32)

    with torch.no_grad():
        torch._export.aot_compile(
            model,
            (input_img, memory),
            # dynamic_shapes={"masked_frames": {0: batch_dim}, "former_attn": {0: batch_dim}},
            options={
                "aot_inductor.output_path": os.path.join(
                    os.getcwd(), "inpainting/single/model_single.so"
                )
            },
        )

    input_img = torch.randn((1, 1, 3, 240, 432), device=device, dtype=torch.float32)
    memory = torch.randn([6, 8, 1, 720, 512], device=device, dtype=torch.float32)

    with torch.no_grad():
        torch._export.aot_compile(
            model,
            (input_img, memory),
            # dynamic_shapes={"masked_frames": {0: batch_dim}, "former_attn": {0: batch_dim}},
            options={
                "aot_inductor.output_path": os.path.join(
                    os.getcwd(), "inpainting/memory/model_memory.so"
                )
            },
        )


def main():
    clone_dstt()
    download_weights()
    compile_models()


if __name__ == "__main__":
    main()
