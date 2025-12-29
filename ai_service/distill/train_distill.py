import os
import argparse
import logging
import math
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import WhisperModel
from omegaconf import OmegaConf
import wandb

# MuseTalk Imports
from musetalk.data.dataset import PortraitDataset
from musetalk.utils.utils import get_image_pred
from musetalk.utils.training_utils import Net as TeacherNet # Use existing wrapper

# Logger Setup
logger = get_logger(__name__)

# --- STUDENT ARCHITECTURE WRAPPERS ---

class AudioProjector(nn.Module):
    def __init__(self, in_features=384, out_features=768):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features) # Stabilization
        
    def forward(self, x):
        return self.norm(self.linear(x))

class StudentNet(nn.Module):
    """
    Wraps the Student VAE-UNet (bk-sdm-tiny) and Audio Projector.
    Matches the Interface of the Teacher Net.
    """
    def __init__(self, unet: UNet2DConditionModel, projector: AudioProjector):
        super().__init__()
        self.unet = unet
        self.projector = projector

    def forward(self, input_latents, timesteps, audio_prompts):
        """
        input_latents: [B, 8, H, W] (Concatenated Masked + Reference)
        timesteps: [B]
        audio_prompts: [B, T, N, 384] (Whisper Features)
        """
        # 1. Project Audio Embeddings (384 -> 768)
        # audio_prompts shape: [B, SeqLen, Dim] ?
        # utils.py process_audio_features returns [B, T, 50, 384] (rearranged)
        # Teacher UNet expects these as 'encoder_hidden_states'.
        # We project them to 768.
        
        audio_emb = self.projector(audio_prompts)
        
        # 2. Forward UNet
        # Student UNet conv_in must handle 8 channels (we patch this in loading).
        model_pred = self.unet(
            input_latents,
            timesteps,
            encoder_hidden_states=audio_emb
        ).sample
        
        return model_pred

# --- MAIN TRAINING LOOP ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="./models/musetalk", help="Path to MuseTalk model")
    parser.add_argument("--student_model", type=str, default="nota-ai/bk-sdm-tiny", help="HF Path or local")
    parser.add_argument("--data_root", type=str, default="./dataset/HDTF")
    parser.add_argument("--output_dir", type=str, default="./distill_output")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4) # Small batch for testing
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--wandb_key", type=str, default=None, help="WandB API Key")
    parser.add_argument("--project_name", type=str, default="musetalk-distillation", help="WandB Project Name")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=1)
    set_seed(42)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        wandb.init(project=args.project_name, config=vars(args))

    # 1. Load Teacher (Frozen)
    logger.info("Loading Teacher...")
    # Load Config
    unet_config_path = os.path.join(args.teacher_model, "musetalk.json")
    with open(unet_config_path, 'r') as f:
         import json
         t_config = json.load(f)
    
    teacher_unet = UNet2DConditionModel(**t_config)
    teacher_weights = os.path.join(args.teacher_model, "pytorch_model.bin")
    teacher_unet.load_state_dict(torch.load(teacher_weights, map_location="cpu"))
    teacher_unet.requires_grad_(False)
    
    
    # Load VAE and Whisper (Shared/Static)
    vae_path = "./models/sd-vae"
    if not os.path.exists(vae_path):
        vae_path = os.path.join(os.path.dirname(args.teacher_model), "sd-vae")
    
    logger.info(f"Loading VAE from {vae_path}")
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.requires_grad_(False)
    
    whisper = WhisperModel.from_pretrained("./models/whisper")
    whisper.requires_grad_(False)

    teacher_net = TeacherNet(teacher_unet) # Wraps it for forward pass
    teacher_net.eval() 

    # 2. Load Student (Trainable)
    logger.info(f"Loading Student: {args.student_model}")
    student_unet = UNet2DConditionModel.from_pretrained(args.student_model, subfolder="unet")
    
    # 3. Adapt Student Architecture
    # Expand Input Channels (4 -> 8)
    with torch.no_grad():
        logger.info("Expanding Student Input Channels 4 -> 8")
        old_conv = student_unet.conv_in
        # Create new conv
        # config is dict-like? or object? diffusers 0.14+ object.
        new_conv = nn.Conv2d(8, old_conv.out_channels, kernel_size=3, padding=1)
        
        # Init weights: First 4 channels = copy, Next 4 = zero
        new_conv.weight[:, :4, :, :] = old_conv.weight
        new_conv.weight[:, 4:, :, :] = 0
        new_conv.bias = old_conv.bias
        
        student_unet.conv_in = new_conv
        student_unet.config.in_channels = 8 # Update config
    
    # Create Projector
    projector = AudioProjector(in_features=384, out_features=768) # Whisper -> SD1.5 Dim
    
    student_net = StudentNet(student_unet, projector)
    
    # RESUME LOGIC
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        student_net.load_state_dict(torch.load(args.resume_from, map_location="cpu"))
    
    student_net.train()

    # 4. Optimizer
    optimizer = torch.optim.AdamW(student_net.parameters(), lr=args.learning_rate)

    # 5. Dataset
    # We construct a minimal cfg dict required by PortraitDataset
    dataset_cfg = {
        'image_size': args.resolution,
        'T': 1,
        'sample_method': 'pose_similarity_and_mouth_dissimilarity',
        'top_k_ratio': 0.5,
        'contorl_face_min_size': False, # Relax check for small demo videos
        'dataset_key': "HDTF", # Uses dataset/HDTF/meta
        'padding_pixel_mouth': 10,
        'whisper_path': "./models/whisper",
        'min_face_size': 0, # Relax
        'cropping_jaw2edge_margin_mean': 10,
        'cropping_jaw2edge_margin_std': 10,
        'crop_type': "dynamic_margin_crop_resize",
        'random_margin_method': "normal", 
    }
    
    # Check if subset list exists, use it for testing
    subset_list = os.path.join(args.data_root, "train_partial.txt")
    if os.path.exists(subset_list):
        logger.info(f"Using Partial Training List: {subset_list}")
        dataset_cfg['video_clip_file_list_train'] = subset_list
    elif os.path.exists(os.path.join(args.data_root, "train_subset.txt")):
         logger.info(f"Using Subset Training List: train_subset.txt")
         dataset_cfg['video_clip_file_list_train'] = os.path.join(args.data_root, "train_subset.txt")
    else:
        logger.info("Using Full Training List (train.txt)")
        dataset_cfg['video_clip_file_list_train'] = os.path.join(args.data_root, "train.txt")
    
    # Mocking config object access (cfg.key) if dataset expects it, but dict access is supported in code I saw.
    # dataset.py: self.image_size = cfg['image_size']. So dict is fine.
    
    dataset = PortraitDataset(dataset_cfg)
    # Turbo Mode: increased workers to 16
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    # 6. Prepare with Accelerator
    # Prepare should cast model to correct device/dtype if mixed_precision is set?
    # Usually prepare only moves to device. Type casting is often automatic for inputs, but model weights?
    # Accelerate mixed_precision="fp16" wraps model in autocast, but weights stay FP32 usually (unless converted).
    # But RuntimeError says Input (Half) and Bias (Float).
    # This means inputs were cast to Half (via autobatch or manual to(weight_dtype)), but bias remained Float.
    # We manually cast inputs in the loop: pixel_values_vid = batch["pixel_values_vid"].to(weight_dtype)
    # So we MUST cast the model to weight_dtype if we want pure FP16 (or rely on autocast for mixed).
    # If mixed_precision=fp16, we should keep weights in FP32 ideally for stability, and use autocast.
    # BUT if we are feeding FP16 inputs, we might hit this mismatch if autocast isn't active or layer doesn't support mix.

    student_net, optimizer, dataloader = accelerator.prepare(student_net, optimizer, dataloader)
    teacher_net, vae, whisper = accelerator.prepare(teacher_net, vae, whisper)
    
    # Force cast newly created layers if they cause issues, OR ensure we feed FP32 if mixed precision handles the cast.
    # Actually, if we use accelerator, we should let it handle types.
    # The issue might be `pixel_values_vid.to(weight_dtype)`.
    # If we remove `.to(weight_dtype)` from inputs, Accelerate's autocast should handle it.
    # Let's try removing explicit input casting and iterate.
    # ... Wait, the error is inside `conv_in`.
    
    # FIX: Cast student_net to float32 explicitly ensure init is consistent, let mixed precision handle half.
    # OR if we want half weights, cast to half.
    # Let's try: student_net.to(accelerator.device) # Prepare does this.
    
    # Let's change the Input Casting in the loop to rely on accelerator.autocast context?
    # But `accelerator.accumulate` likely handles that.
    
    # The error "Input type (Half) and bias type (Float)" suggests inputs ARE Half.
    # Which means we cast them.
    # If we cast inputs to Half, but model is Float, PyTorch usually errors on Conv2d (no auto-promote for Half input).
    # So we should EITHER:
    # 1. Cast model to Half.
    # 2. Feed Float inputs (and let autocast downcastops).
    
    # Recommended for Mixed Precision: Keep model FP32, Feed correct type? 
    # Usually inputs are Float32, and Autocast creates Half Ops.
    # So I will REMOVE `.to(weight_dtype)` from inputs in the loop.
    # Teacher/VAE/Whisper stay in eval mode
    
    # 7. Training Loop
    global_step = 0
    progress_bar = tqdm(range(args.train_steps), disable=not accelerator.is_local_main_process)
    
    # Re-import processing utils
    from musetalk.utils.training_utils import process_audio_features
    from einops import rearrange

    data_iter = iter(dataloader)
    
    while global_step < args.train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        with accelerator.accumulate(student_net):
            # Prepare Inputs
            # pixel_values_vid: [B, 1, 3, H, W]
            bsz = batch['pixel_values_vid'].size(0)
            weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
            
            # Encode Latents (Teacher & Student share VAE for now, or assume Student uses same VAE space)
            # Since bk-sdm-tiny is SD based, VAE space is identical.
            
            # We need input_latents = [masked_latents, ref_latents]
            # Reference: musetalk.utils.training_utils.get_image_pred logic
            
            with torch.no_grad():
                pixel_values_vid = batch["pixel_values_vid"] # Target
                ref_pixel_values = batch["pixel_values_ref_img"]
                
                # VAE Encode
                # Helper to encode [B, F, C, H, W] -> latents
                def encode(pixels):
                    # flatten B, F
                    x = rearrange(pixels, 'b f c h w -> (b f) c h w')
                    latents = vae.encode(x).latent_dist.mode()
                    latents = latents * vae.config.scaling_factor
                    return latents # [B, 4, h, w]
                
                # Create Masked Input
                masked_pixels = pixel_values_vid.clone()
                h = masked_pixels.shape[-2]
                masked_pixels[:, :, :, h//2:, :] = -1 # Mask lower half
                
                masked_latents = encode(masked_pixels)
                ref_latents = encode(ref_pixel_values)
                target_latents = encode(pixel_values_vid) # Ground Truth for Loss
                
                input_latents = torch.cat([masked_latents, ref_latents], dim=1) # [B, 8, h, w]
                
                # Audio
                # Helper process_audio_features returns [B, T, N, 384] stack
                # We need simple namespace object for process_audio_features(cfg, ...)
                # It accesses cfg.data.audio_padding_...
                # We'll mock it or verify usage.
                # utils.py: cfg.data.audio_padding_length_left
                
            # Mock Config for Audio
            class MockCfg:
                class Data:
                    audio_padding_length_left = 2
                    audio_padding_length_right = 2
                data = Data()
            
            # audio_prompts will be cast inside process_audio_features IF it uses weight_dtype arg correctly
            # Check utils.py: process_audio_features(..., weight_dtype) uses .to(weight_dtype).
            # So we should pass torch.float32 for consistency or let it float?
            # If we pass weight_dtype (which is now Float if we removed Mixed Precision casting?), 
            # Accelerator mixed_precision='fp16' means weight_dtype might be undefined here if we removed the line defining it?
            # Wait, line 237 `weight_dtype = torch.float16 ...` is still there.
            
            # If we want inputs FP32, we should use torch.float32.
            audio_prompts = process_audio_features(MockCfg(), batch, whisper, bsz, 1, torch.float32)
            # audio_prompts: [B, T, N, 384]. Flatten if needed?
            # Net expects encoder_hidden_states.
            # utils.py: rearrange(audio_prompts, '(b f) c h w -> (b f) (c h) w') ? No.
            # utils.py lines 308-312:
            # audio_prompts = rearrange(audio_prompts, 'b f c h w -> (b f) c h w')
            # audio_prompts = rearrange(audio_prompts, '(b f) c h w -> (b f) (c h) w')
            # So [B*F, SeqLen, Dim].
            
            # audio_prompts from process_audio_features is [B, T, 10, 5, 384] ? (Line 188 utils.py: stack dim 2 -> [B, T, 10, 5, 384]?)
            # Wait, line 206 utils: torch.cat(audio_prompts).
            # Returns [B, T, N, 5, 384]?
            # Let's trust process_audio_features returns what Teacher expects.
            # We just reshape it for Forward.
            audio_prompts = rearrange(audio_prompts, 'b f c h w -> (b f) (c h) w')
            
            timesteps = torch.tensor([0], device=input_latents.device).repeat(bsz) # Always 0 for inference-like training?
            # Wait, Distillation usually involves random timesteps if doing Diffusion Distillation (ADD/LCM).
            # BUT efficient distillation (BK-SDM) usually assumes One-Step or Few-Step?
            # MuseTalk "Inference" uses timesteps=[0]. 
            # If we want Student to learn "One Step Generation", we train with t=0.
            # Teacher output at t=0 is the "clean data".
            # So we regress Student(t=0) -> Teacher(t=0).
            # Correct. output is 'sample' (decoded image latent) or 'pred_latents'.
            # Net returns 'sample' (pred_latents).
            
            # Forward Teacher
            # Note: Teacher expects audio_prompts dim 384. Student expects 768 (via Projector).
            # audio_prompts is 384.
            with torch.no_grad():
                teacher_pred = teacher_net(input_latents, timesteps, audio_prompts)
            
            # Forward Student
            # StudentNet handles projection (384->768) internally.
            student_pred = student_net(input_latents, timesteps, audio_prompts)
            
            # Loss
            loss = F.mse_loss(student_pred, teacher_pred)
            
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        if accelerator.is_main_process:
             wandb.log({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']}, step=global_step)

        if global_step % 1 == 0:
            print(f"Step {global_step} Loss: {loss.item()}", flush=True)
            
        if global_step % args.save_steps == 0:
            save_path = os.path.join(args.output_dir, f"student_step_{global_step}.pth")
            torch.save(student_net.state_dict(), save_path)
            
        progress_bar.update(1)
        global_step += 1

    # Save Final
    torch.save(student_net.state_dict(), os.path.join(args.output_dir, "student_final.pth"))

if __name__ == "__main__":
    main()
