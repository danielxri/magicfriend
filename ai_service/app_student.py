
import sys
import torch
from pathlib import Path
import os
import shutil
import subprocess
import glob
import cv2
import pickle
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uuid
import imageio_ffmpeg
import time

# Add MuseTalk to path
sys.path.append(str(Path(__file__).parent / "MuseTalk"))

# MuseTalk imports
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from omegaconf import OmegaConf
from transformers import WhisperModel
from diffusers import AutoencoderKL, UNet2DConditionModel

# Student Model Imports
from musetalk.models.student import StudentNet, AudioProjector

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model Paths (Relative to ai_service/MuseTalk)
MUSE_DIR = Path(__file__).parent / "MuseTalk"
UNET_CONFIG = MUSE_DIR / "models/musetalkV15/musetalk.json"
UNET_MODEL_PATH = MUSE_DIR / "models/musetalkV15/unet.pth"
VAE_TYPE = "sd-vae" # or "gan"
WHISPER_DIR = MUSE_DIR / "models/whisper" 

# Student Checkpoint
STUDENT_CHECKPOINT = MUSE_DIR / "models/musetalk/student_prod.pth"

# Global models
vae = None
student_net = None # Replaces unet
pe = None # Not needed for Student, likely
whisper = None
audio_processor = None
face_parser = None

def load_models():
    global vae, student_net, pe, whisper, audio_processor, face_parser
    
    if vae is not None:
        return

    print("Loading Student models...")
    
    # Load VAE, UNet, PE (Teacher Components we might reuse parts of)
    # We use load_all_model mainly for VAE and PE if needed. 
    # But Student doesn't use PE.
    # We still need VAE.
    
    print("Loading VAE and PE from load_all_model...")
    vae, _, pe = load_all_model(
        unet_model_path=str(UNET_MODEL_PATH), # Dummy path for UNet
        vae_type=VAE_TYPE,
        unet_config=str(UNET_CONFIG),
        device=device,
        vae_model_path=str(MUSE_DIR / "models" / VAE_TYPE)
    )

    # Move VAE/PE to device
    weight_dtype = torch.float16
    pe = pe.to(device, dtype=weight_dtype) # Might not be used
    vae.vae = vae.vae.to(device)

    # Compile VAE Decoder (Critical for Loop Speed)
    print("Compiling VAE Decoder...")
    try:
        vae.vae.decoder = torch.compile(vae.vae.decoder, mode="max-autotune")
    except Exception as e:
        print(f"Warning: VAE compile failed: {e}")
    
    # --- LOAD STUDENT MODEL which replaces Teacher UNet ---
    print(f"Loading Student Model from {STUDENT_CHECKPOINT}...")
    
    # 1. Base UNet (bk-sdm-tiny structure)
    base_unet = UNet2DConditionModel.from_pretrained("nota-ai/bk-sdm-tiny", subfolder="unet")
    
    # Patch conv_in channels: 4 -> 8
    old_conv = base_unet.conv_in
    new_conv = torch.nn.Conv2d(8, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding)
    base_unet.conv_in = new_conv
    
    # Initialize Projector
    projector = AudioProjector(in_features=384, out_features=768)
    
    # Wrap in StudentNet
    student_net = StudentNet(base_unet, projector)
    
    # Load Weights
    if os.path.exists(STUDENT_CHECKPOINT):
        checkpoint = torch.load(STUDENT_CHECKPOINT, map_location="cpu")
        student_net.load_state_dict(checkpoint)
        print("Student Checkpoint Loaded.")
    else:
        print(f"CRITICAL ERROR: Student checkpoint not found at {STUDENT_CHECKPOINT}")
        print("Falling back to uninitialized student model (GARBAGE OUTPUT WARNING)")
    
    # Convert Student to FP16 and Device
    student_net = student_net.to(dtype=weight_dtype)
    student_net = student_net.to(device)
    student_net.eval()
    
    # Pre-compile Student (Crucial for Speed on GB10)
    # print("Compiling Student UNet with torch.compile...")
    # try:
    #     # Compile the UNet inside StudentNet
    #     # Note: compiling the whole StudentNet might issue if Projector is involved, 
    #     # but usually compiling the heavy UNet part is enough/safer.
    #     student_net.unet = torch.compile(student_net.unet, mode="max-autotune") # reduce-overhead or max-autotune
    # except Exception as e:
    # 3. Aggressive Compilation (End-to-End, Reduce Overhead for Latency)
    print("Compiling StudentNet (End-to-End) with mode='reduce-overhead'...")
    try:
        # Compiling the wrapper optimizes Projector + UNet + Data movement
        student_net = torch.compile(student_net, mode="reduce-overhead")
    except Exception as e:
        print(f"Warning: torch.compile failed: {e}")


    # Load Whisper
    print("Loading Whisper...")
    audio_processor = AudioProcessor(feature_extractor_path=str(WHISPER_DIR))
    whisper = WhisperModel.from_pretrained(str(WHISPER_DIR))
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Load Face Parser
    print("Loading Face Parser...")
    face_parser = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    
    print("Student Models loaded successfully.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

from fastapi.staticfiles import StaticFiles
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

class TalkRequest(BaseModel):
    text: str
    sessionId: str

@app.on_event("startup")
async def startup_event():
    try:
        # Create symlink for models
        if not os.path.exists("models") and (MUSE_DIR / "models").exists():
            try:
                os.symlink(MUSE_DIR / "models", "models")
                print(f"Created symlink ./models -> {MUSE_DIR}/models")
            except Exception as e:
                print(f"Failed to create models symlink: {e}")

        load_models()
    except Exception as e:
        print(f"Failed to load models at startup: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": vae is not None,
        "mode": "student_distilled"
    }

@app.post("/generate_avatar")
async def generate_avatar(
    audio: UploadFile = File(None),
    text: str = Form(None),
    image: UploadFile = File(None),
    sessionId: str = Form(...)
):
    if not audio and not text:
        raise HTTPException(status_code=400, detail="Either audio or text is required")
    
    job_id = f"{sessionId}_{uuid.uuid4().hex[:8]}"
    
    image_path = None
    if image:
        image_path = UPLOAD_DIR / f"{job_id}_input.png"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
    
    if not image_path or not image_path.exists():
         raise HTTPException(status_code=400, detail="Avatar image is required")

    audio_path = UPLOAD_DIR / f"{job_id}_input.wav"
    
    if audio:
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
    elif text:
        pass # Todo TTS
        
    if not audio_path.exists():
         raise HTTPException(status_code=400, detail="Audio required (TTS not implemented)")

    try:
        output_video = run_inference(str(image_path), str(audio_path), job_id)
        return {"status": "success", "video_url": f"/outputs/{output_video.name}"}
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class Timer:
    def __init__(self, name):
        self.name = name
        self.start = 0
    def __enter__(self):
        self.start = time.time()
        print(f"[LATENCY] Start {self.name}...")
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        end = time.time()
        duration = (end - self.start) * 1000
        print(f"[LATENCY] End {self.name}: {duration:.2f}ms")

LANDMARKS_CACHE = {}

from musetalk.utils.blending import get_image_prepare_material, get_image_blending

def run_inference(image_path, audio_path, job_id):
    try:
        session_id = job_id.split('_')[0]
    except:
        session_id = job_id
    
    weight_dtype = torch.float16

    with Timer("Total Inference"):
        fps = 25
        batch_size = 8 # Safety conservative for Student first run
        
        with Timer("Whisper Feature Extraction"):
            # Ensure safe audio loading (retry logic or ffmpeg check?)
            # Assuming audio_processor works if ffmpeg path is correct on system
            try:
                whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
            except TypeError:
                # Likely ffmpeg/load error, try to convert audio first?
                # Or just raise
                raise Exception("Failed to process audio file. Check format.")
        
        with Timer("Whisper Chunking"):
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features, 
                device, 
                weight_dtype, 
                whisper, 
                librosa_length,
                fps=fps
            )
        
        input_img_list = [image_path]
        
        global LANDMARKS_CACHE
        coord_list = None
        frame_list = None
        
        if session_id in LANDMARKS_CACHE:
             print(f"[CACHE] Hit for session {session_id}")
             coord_list, frame_list = LANDMARKS_CACHE[session_id]
        else:
             print(f"[CACHE] Miss for session {session_id}, running detection")
             with Timer("Face Detection"):
                bbox_shift = 0 
                # Catch emptyface
                try:
                    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                except Exception as e:
                    raise Exception(f"Face detection failed: {e}")
                
                LANDMARKS_CACHE[session_id] = (coord_list, frame_list)
        
        if not coord_list or len(coord_list) == 0:
             raise Exception("No face detected")
             
        bbox = coord_list[0]
        if bbox == coord_placeholder:
            raise HTTPException(status_code=400, detail="No face detected in the provided avatar image.")
        frame = frame_list[0]
        x1, y1, x2, y2 = bbox
        
        y2 = y2 + 10 
        y2 = min(y2, frame.shape[0])
        
        with Timer("VAE Encoding"):
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list = [latents]
        
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list,
            batch_size=batch_size,
            delay_frame=0,
            device=device
        )
        
        res_frame_list = []
        
        with Timer("Student UNet Inference Loop"):
            for i, (whisper_batch, latent_batch) in enumerate(gen):
                # Student Interface:
                # Inputs: (latents, timesteps, audio_prompts)
                
                # audio_prompts should be raw whisper features from generator [B, T, 384] ??
                # datagen yields:
                # whisper_batch: [B, W, 384] (Window size 50) ??
                
                # Checking inference_student.py logic:
                # pred_latents = student_net(latent_batch, timesteps, whisper_batch)
                
                latent_batch = latent_batch.to(dtype=weight_dtype)
                whisper_batch = whisper_batch.to(dtype=weight_dtype) # [B, 50, 384]
                
                timesteps = torch.tensor([0], device=device)
                
                # INFERENCE CALL
                with Timer(f"Batch {i} UNet"):
                    pred_latents = student_net(latent_batch, timesteps, whisper_batch)
                
                with Timer(f"Batch {i} VAE Decode"):
                    recon = vae.decode_latents(pred_latents)
                
                for res_frame in recon:
                    res_frame_list.append(res_frame)



def match_color(target_img, source_img):
    # Convert to LAB (Lightness flows better than RGB for color transfer)
    # Simple mean/std transfer
    try:
        source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype("float32")
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype("float32")
        
        # Compute stats
        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)
        
        source_mean = source_mean.flatten()
        source_std = source_std.flatten()
        target_mean = target_mean.flatten()
        target_std = target_std.flatten()
        
        # Avoid division by zero
        target_std[target_std == 0] = 1.0
        
        # Transfer
        # (Target - TargetMean) * (SourceStd / TargetStd) + SourceMean
        target_lab = (target_lab - target_mean) * (source_std / target_std) + source_mean
        
        # Clip
        target_lab = np.clip(target_lab, 0, 255).astype("uint8")
        
        # Convert back
        return cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR)
    except:
        return target_img

def run_inference(image_path, audio_path, job_id):
    try:
        session_id = job_id.split('_')[0]
    except:
        session_id = job_id
    
    weight_dtype = torch.float16

    with Timer("Total Inference"), torch.no_grad():
        fps = 25
        # batch_size = 48 # Too slow (~65s), possibly memory thrashing/bandwidth saturation.
        # batch_size = 12 
        batch_size = 16 # Power of 2, aligned for Tensor Cores
        
        with Timer("Whisper Feature Extraction"):
            try:
                whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
            except TypeError:
                raise Exception("Failed to process audio file. Check format.")
        
        with Timer("Whisper Chunking"):
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features, 
                device, 
                weight_dtype, 
                whisper, 
                librosa_length,
                fps=fps
            )
        
        input_img_list = [image_path]
        
        global LANDMARKS_CACHE
        coord_list = None
        frame_list = None
        
        if session_id in LANDMARKS_CACHE:
             print(f"[CACHE] Hit for session {session_id}")
             coord_list, frame_list = LANDMARKS_CACHE[session_id]
        else:
             print(f"[CACHE] Miss for session {session_id}, running detection")
             with Timer("Face Detection"):
                bbox_shift = 0 
                try:
                    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                except Exception as e:
                    raise Exception(f"Face detection failed: {e}")
                
                LANDMARKS_CACHE[session_id] = (coord_list, frame_list)
        
        if not coord_list or len(coord_list) == 0:
             raise Exception("No face detected")
             
        bbox = coord_list[0]
        if bbox == coord_placeholder:
            raise HTTPException(status_code=400, detail="No face detected in the provided avatar image.")
        frame = frame_list[0]
        x1, y1, x2, y2 = bbox
        
        y2 = y2 + 10 
        y2 = min(y2, frame.shape[0])
        
        with Timer("VAE Encoding"):
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list = [latents]
        
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list,
            batch_size=batch_size,
            delay_frame=0,
            device=device
        )
        
        res_frame_list = []
        
        with Timer("Student UNet Inference Loop"):
            for i, (whisper_batch, latent_batch) in enumerate(gen):
                latent_batch = latent_batch.to(dtype=weight_dtype)
                whisper_batch = whisper_batch.to(dtype=weight_dtype) 
                
                timesteps = torch.tensor([0], device=device)
                
                pred_latents = student_net(latent_batch, timesteps, whisper_batch)
                
                recon = vae.decode_latents(pred_latents)
                
                for res_frame in recon:
                    res_frame_list.append(res_frame)


        # Zero-Copy Video Encoding Pipeline
        output_path = OUTPUT_DIR / f"{job_id}.mp4"
        ffmpeg_exe = "/usr/bin/ffmpeg" 

        height, width, _ = frame.shape
        
        cmd = [
            ffmpeg_exe, "-y", "-v", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24", 
            "-r", str(fps),
            "-i", "-", 
            "-i", str(audio_path),
            "-vcodec", "libx264", 
            "-preset", "ultrafast", 
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]

        print(f"[FFMPEG] Streaming to: {output_path}")
        
        video_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        
        # Capture the original crop for color matching
        ori_crop = frame[y1:y2, x1:x2]
        
        with Timer("Face Blending (GPU-Piped)"):
            mask_array, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], mode='jaw', fp=face_parser)
            
            for i, res_frame in enumerate(res_frame_list):
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                    
                    # COLOR MATCHING
                    # Match res_frame (generated) to ori_crop (original)
                    # res_frame = match_color(res_frame, ori_crop) # DISABLED FOR SPEED PROOF


                    
                except:
                    continue
                
                combine_frame = get_image_blending(frame, res_frame, [x1, y1, x2, y2], mask_array, crop_box)
                
                try:
                    video_process.stdin.write(combine_frame.tobytes())
                except Exception as e:
                    print(f"FFmpeg Pipe Write Error: {e}")
                    break
        
        video_process.stdin.close()
        video_process.wait()
        
        if video_process.returncode != 0:
             print(f"FFmpeg failed with code {video_process.returncode}")
        
        return output_path

