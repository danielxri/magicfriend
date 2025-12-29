import torch
from diffusers import UNet2DConditionModel

def check_model(model_id):
    print(f"\nChecking {model_id}...")
    try:
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        params = sum(p.numel() for p in unet.parameters()) / 1e6
        print(f"  - Parameters: {params:.2f}M")
        print(f"  - In Channels: {unet.config.in_channels}")
        print(f"  - Sample Size: {unet.config.sample_size}")
        print(f"  - Block Out Channels: {unet.config.block_out_channels}")
        print(f"  - Layers per Block: {unet.config.layers_per_block}")
        print(f"  - Cross Attention Dim: {unet.config.cross_attention_dim}")
        return True
    except Exception as e:
        print(f"  - Error loading: {e}")
        return False

models = [
    "nota-ai/bk-sdm-tiny",
    "segmind/tiny-sd",
    "runwayml/stable-diffusion-v1-5" # Baseline
]

for m in models:
    check_model(m)
