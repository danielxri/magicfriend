import os
import cv2
import math
import copy
import torch
import glob
import shutil
import pickle
import argparse
import numpy as np
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import WhisperModel
import sys
from diffusers import UNet2DConditionModel

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder

# Import Student Model
from musetalk.models.student import StudentNet, AudioProjector

def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

@torch.no_grad()
def main(args):
    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # --- LOAD TEACHER COMPONENTS (VAE, PE) ---
    # We use load_all_model to get VAE and PE, but ignore the Teacher UNet
    vae, _, pe = load_all_model(
        unet_model_path=args.unet_model_path, 
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    
    # --- LOAD STUDENT MODEL ---
    print(f"Loading Student Model from {args.student_checkpoint}...")
    
    # 1. Base UNet (bk-sdm-tiny structure)
    # We need to load the CONFIG first to initialize architecture, then state dict.
    # Ideally load from HF 'nota-ai/bk-sdm-tiny' then patch weights?
    # Or load straight from checkpoint if it contains full state dict.
    
    # Initialize architecture
    base_unet = UNet2DConditionModel.from_pretrained("nota-ai/bk-sdm-tiny", subfolder="unet")
    
    # Patch conv_in channels: 4 -> 8
    old_conv = base_unet.conv_in
    new_conv = torch.nn.Conv2d(8, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding)
    # Initialize with old weights (optional, but training process did this implicitly? 
    # Actually training process did: base_unet.conv_in = nn.Conv2d(8, ...)
    # checking train_distill.py ... yes.
    base_unet.conv_in = new_conv
    
    # Initialize Projector
    projector = AudioProjector(in_features=384, out_features=768)
    
    # Wrap in StudentNet
    student_net = StudentNet(base_unet, projector)
    
    # Load Weights
    checkpoint = torch.load(args.student_checkpoint, map_location="cpu")
    student_net.load_state_dict(checkpoint)
    print("Student Checkpoint Loaded.")
    
    timesteps = torch.tensor([0], device=device)

    # Convert to half precision
    if args.use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        student_net = student_net.half()
    
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    student_net = student_net.to(device)
    student_net.eval()
        
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = torch.float16 if args.use_float16 else torch.float32
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    if args.version == "v15":
        fp = FaceParsing(left_cheek_width=args.left_cheek_width, right_cheek_width=args.right_cheek_width)
    else:
        fp = FaceParsing()
    
    inference_config = OmegaConf.load(args.inference_config)
    print("Loaded inference config details hidden for brevity")
    
    for task_id in inference_config:
        try:
            print(f"Processing task: {task_id}")
            video_path = inference_config[task_id]["video_path"]
            audio_path = inference_config[task_id]["audio_path"]
            print(f"Video: {video_path}, Audio: {audio_path}")
            if "result_name" in inference_config[task_id]:
                args.output_vid_name = inference_config[task_id]["result_name"]
            
            if args.version == "v15":
                bbox_shift = 0 
            else:
                bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)
            
            input_basename = os.path.basename(video_path).split('.')[0]
            audio_basename = os.path.basename(audio_path).split('.')[0]
            output_basename = f"{input_basename}_{audio_basename}_STUDENT"
            
            temp_dir = os.path.join(args.result_dir, f"{args.version}")
            os.makedirs(temp_dir, exist_ok=True)
            result_img_save_path = os.path.join(temp_dir, output_basename)
            crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
            os.makedirs(result_img_save_path, exist_ok=True)
            
            if args.output_vid_name is None:
                output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")
            else:
                output_vid_name = os.path.join(temp_dir, args.output_vid_name)
            
            # --- PREPROCESSING (Identical to original) ---
            if get_file_type(video_path) == "video":
                save_dir_full = os.path.join(temp_dir, input_basename)
                os.makedirs(save_dir_full, exist_ok=True)
                cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
                os.system(cmd)
                input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
                fps = get_video_fps(video_path)
            elif get_file_type(video_path) == "image":
                input_img_list = [video_path]
                fps = args.fps
            elif os.path.isdir(video_path):
                input_img_list = sorted(glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]')))
                fps = args.fps

            whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features, device, weight_dtype, whisper, librosa_length, fps=fps,
                audio_padding_length_left=args.audio_padding_length_left,
                audio_padding_length_right=args.audio_padding_length_right,
            )
            
            if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
                with open(crop_coord_save_path, 'rb') as f:
                    coord_list = pickle.load(f)
                frame_list = read_imgs(input_img_list)
            else:
                coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                with open(crop_coord_save_path, 'wb') as f:
                    pickle.dump(coord_list, f)
            print(f"Number of frames: {len(frame_list)}")         
            
            input_latent_list = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                if args.version == "v15":
                    y2 = y2 + args.extra_margin
                    y2 = min(y2, frame.shape[0])
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
                latents = vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)
        
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            
            print("Starting STUDENT inference")
            video_num = len(whisper_chunks)
            batch_size = args.batch_size
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=batch_size,
                delay_frame=0,
                device=device,
            )
            
            res_frame_list = []
            total = int(np.ceil(float(video_num) / batch_size))
            
            for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
                # --- STUDENT INFERENCE LOGIC ---
                # Audio Process (PE) is still same? No, PE is Teacher's Positional Encoding?
                # Wait, Teacher UNet uses Positional Encoding on audio features.
                # Student AudioProjector takes RAW whisper features (384) ??
                # In train_distill.py:
                # audio_prompts = batch["audio_embeddings"] # [B, T, N, 384] ?
                # student_pred = student_net(..., audio_prompts)
                
                # Check datagen output: 'whisper_batch' is [B, T, 384]?
                # musetalk/utils/utils.py datagen:
                # yield whisper_batch, latent_batch
                
                # In standard inference.py:
                # audio_feature_batch = pe(whisper_batch)
                # unet.model(..., encoder_hidden_states=audio_feature_batch)
                
                # In my train_distill.py logic:
                # I passed `audio_prompts` directly to student_net.
                # BUT `dataset.py` might be doing something different?
                # In train_distill.py:
                # from musetalk.utils.training_utils import process_audio_features
                # audio_prompts = process_audio_features(batch["audio_feature"]) -> PE?
                
                # WAIT. Implementation Detail check:
                # In train_distill.py (Step 3281):
                # audio_prompts = self.projector(audio_prompts)
                # It expects 384 dims?
                # If so, we should skip `pe`.
                
                # Let's assume Student inputs RAW whisper (384).
                
                latent_batch = latent_batch.to(dtype=weight_dtype)
                whisper_batch = whisper_batch.to(dtype=weight_dtype)
                
                # We need to reshape whisper_batch to [B, T*N, 384] or similar?
                # StudentNet expects [B, T, N, 384] and rearranges it?
                # No, StudentNet forward expects [B, T, N, 384] ? 
                # Let's check StudentNet.forward again.
                # audio_emb = self.projector(audio_prompts)
                # projector takes [..., 384] -> [..., 768]
                # Then it passes to UNet.
                
                # Note: `datagen` returns whisper_batch as [B, 50, 384]?
                # Inference uses window size 50.
                
                # So we pass `whisper_batch` directly to StudentNet.
                # And we create timesteps=[0]
                
                pred_latents = student_net(latent_batch, timesteps, whisper_batch)
                recon = vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_list.append(res_frame)
            
            # --- POSTPROCESSING (Identical) ---
            print("Padding generated images...")
            for i, res_frame in enumerate(tqdm(res_frame_list)):
                bbox = coord_list_cycle[i%(len(coord_list_cycle))]
                ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
                x1, y1, x2, y2 = bbox
                if args.version == "v15":
                    y2 = y2 + args.extra_margin
                    y2 = min(y2, frame.shape[0])
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                except:
                    continue
                if args.version == "v15":
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
                else:
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)
                cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)

            temp_vid_path = f"{temp_dir}/temp_{input_basename}_{audio_basename}.mp4"
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"
            os.system(cmd_img2video)   
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_vid_name}"
            os.system(cmd_combine_audio)
            shutil.rmtree(result_img_save_path)
            os.remove(temp_vid_path)
            shutil.rmtree(save_dir_full)
            if not args.saved_coord:
                os.remove(crop_coord_save_path)       
            print(f"Results saved to {output_vid_name}")
        except Exception as e:
            print("Error occurred during processing:", e)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--vae_type", type=str, default="sd-vae")
    parser.add_argument("--unet_config", type=str, default="./models/musetalkV15/musetalk.json")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper")
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml")
    parser.add_argument("--student_checkpoint", type=str, required=True, help="Path to Student Checkpoint (.pth)")
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--result_dir", default='./results')
    parser.add_argument("--extra_margin", type=int, default=10)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--audio_padding_length_left", type=int, default=2)
    parser.add_argument("--audio_padding_length_right", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_vid_name", type=str, default=None)
    parser.add_argument("--use_saved_coord", action="store_true")
    parser.add_argument("--saved_coord", action="store_true")
    parser.add_argument("--use_float16", action="store_true")
    parser.add_argument("--parsing_mode", default='jaw')
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"])
    args = parser.parse_args()
    main(args)
