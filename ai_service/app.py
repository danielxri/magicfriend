print("DEBUG: LOADED APP V2")

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
unet = None # Teacher UNet
pe = None 
whisper = None
audio_processor = None
face_parser = None

def load_models():
    global vae, unet, pe, whisper, audio_processor, face_parser
    
    if vae is not None:
        return

    print("Loading Student models...")
    
    # Load VAE, UNet, PE (Teacher Components we might reuse parts of)
    # We use load_all_model mainly for VAE and PE if needed. 
    # But Student doesn't use PE.
    # We still need VAE.
    
    print("Loading Teacher VAE, UNet, PE from load_all_model...")
    vae, unet, pe = load_all_model(
        unet_model_path=str(UNET_MODEL_PATH), 
        vae_type=VAE_TYPE,
        unet_config=str(UNET_CONFIG),
        device=device,
        vae_model_path=str(MUSE_DIR / "models" / VAE_TYPE)
    )

    # Move to device
    weight_dtype = torch.float16
    pe = pe.to(device, dtype=weight_dtype) 
    vae.vae = vae.vae.to(device, dtype=weight_dtype) # VAE FP16 Speedup
    unet.model = unet.model.to(device, dtype=weight_dtype) 
    
    # Enable XFormers (Memory Efficient Attention) - Huge Speedup on NVIDIA
    try:
        unet.model.enable_xformers_memory_efficient_attention()
        print("XFormers Enabled.")
    except Exception as e:
        print(f"XFormers not available: {e}")

    # Load Whisper
    print("Loading Whisper...")
    audio_processor = AudioProcessor(feature_extractor_path=str(WHISPER_DIR))
    whisper = WhisperModel.from_pretrained(str(WHISPER_DIR))
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Load Face Parser
    print("Loading Face Parser...")
    face_parser = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    
    # Load Student Model
    global student_net
    print(f"Loading Student Model from {STUDENT_CHECKPOINT}...")
    student_net = StudentNet()
    student_state_dict = torch.load(STUDENT_CHECKPOINT, map_location=device)
    student_net.load_state_dict(student_state_dict)
    student_net.to(device, dtype=weight_dtype)
    student_net.eval()
    
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

    # Return the stream URL immediately
    return {"status": "success", "video_url": f"/stream/{job_id}"}

@app.get("/stream/{job_id}")
def stream_avatar(job_id: str):
    print(f"[DEBUG] Received stream request for {job_id}")
    image_path = UPLOAD_DIR / f"{job_id}_input.png"
    audio_path = UPLOAD_DIR / f"{job_id}_input.wav"
    
    if not image_path.exists() or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Session expired or not found")
        
    from fastapi.responses import StreamingResponse
    try:
        # run_inference returns a generator function
        stream_gen = run_inference(str(image_path), str(audio_path), job_id)
        
        return StreamingResponse(
            stream_gen(), 
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"inline; filename={job_id}.mp4",
                "X-Accel-Buffering": "no", # Disable Nginx buffering if present
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        print(f"Stream error: {e}")
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

def custom_datagen(
    whisper_chunks,
    vae_encode_latents,
    batch_size=8,
    delay_frame=0,
    device="cuda:0",
):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            wb = torch.stack(whisper_batch)
            lb = torch.cat(latent_batch, dim=0)
            yield wb, lb, batch_size
            whisper_batch, latent_batch  = [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        current_len = len(latent_batch)
        if current_len < batch_size:
            pad_len = batch_size - current_len
            # Pad by duplicating the last element
            last_w = whisper_batch[-1]
            last_l = latent_batch[-1]
            for _ in range(pad_len):
                whisper_batch.append(last_w)
                latent_batch.append(last_l)
        
        wb = torch.stack(whisper_batch)
        lb = torch.cat(latent_batch, dim=0)
        yield wb.to(device), lb.to(device), current_len

# Removed duplicate/broken run_inference

def match_color(target_img, source_img):
    # ... (Keep existing match_color)
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
        fps = 21 # Lowered to 21 to ensure Buffer Safety
        batch_size = 48 # Optimized for GB10 (Turbo)
        
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
        
        # --- STREAMING SETUP ---
        # Output to STDOUT (pipe:1) for HTTP Streaming
        ffmpeg_exe = "/usr/bin/ffmpeg" 
        height, width, _ = frame.shape
        
        # Fragmented MP4 for low-latency streaming
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
            "-tune", "zerolatency", # Vital for realtime
            "-pix_fmt", "yuv420p",
            "-g", str(fps), # Keyframe every 1 second (CRITICAL for Streaming Recovery)
            "-map", "0:v", 
            "-map", "1:a", 
            "-c:a", "aac", 
            "-shortest",   
            "-movflags", "frag_keyframe+empty_moov+default_base_moof", # Enhanced fragmentation
            "-frag_duration", "100000", # Force fragment every ~100ms
            "-f", "mp4",
            "pipe:1" 
        ]

        # print(f"[FFMPEG] Streaming to STDOUT")
        
        # Use a larger buffer for pipe to avoid blocking? Default is usually fine.
        video_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        
        # 2. Prepare Face Blending Material BEFORE the loop
        ori_crop = frame[y1:y2, x1:x2]
        with Timer("Face Blending Prep"):
            mask_array, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], mode='jaw', fp=face_parser)
        
        # Generator function for StreamingResponse
        def stream_generator():
            print(f"[DEBUG] Inside stream_generator for {job_id}")
            print(f"[DEBUG] Generator Chunks: {len(whisper_chunks)}")
            import queue
            import threading
            
            output_queue = queue.Queue()
            
            def reader_thread():
                try:
                    while True:
                        chunk = video_process.stdout.read(4096)
                        if not chunk:
                            break
                        output_queue.put(chunk)
                except Exception as e:
                    print(f"Reader thread error: {e}")
                finally:
                    output_queue.put(None) # Signal end
            
            t = threading.Thread(target=reader_thread, daemon=True)
            t.start()
            
            try:
                # 3. Inference & Streaming Loop
                gen = custom_datagen(
                    whisper_chunks=whisper_chunks,
                    vae_encode_latents=input_latent_list,
                    batch_size=batch_size,
                    delay_frame=0,
                    device=device
                )
                
                for i, (whisper_batch, latent_batch, current_len) in enumerate(gen):
                    try:
                        print(f"[DEBUG] Batch {i} Processing... (Valid: {current_len}/{len(whisper_batch)})")
                        
                        latent_batch = latent_batch.to(dtype=weight_dtype)
                        whisper_batch = whisper_batch.to(dtype=weight_dtype) 
                        
                        # Teacher Logic: PE -> UNet
                        audio_feature_batch = pe(whisper_batch).to(dtype=weight_dtype)
                        timesteps = torch.tensor([0], device=device, dtype=weight_dtype)
                        
                        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                        
                        # Slice Valid Results (Remove Padding)
                        pred_latents = pred_latents[:current_len]
                        
                        # VAE Decode
                        recon = vae.decode_latents(pred_latents)
                    except Exception as e:
                        print(f"CRITICAL INFERENCE ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    # B. Blending & Streaming (Per Batch)
                    for res_frame in recon:
                        try:
                            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                            # Note: match_color disabled for speed
                            combine_frame = get_image_blending(frame, res_frame, [x1, y1, x2, y2], mask_array, crop_box)
                            video_process.stdin.write(combine_frame.tobytes())
                        except Exception as e:
                             print(f"Streaming write error: {e}")
                             continue
                    
                    # C. Yield available bytes from Queue
                    while not output_queue.empty():
                        try:
                            chunk = output_queue.get_nowait()
                            if chunk is None:
                                break
                            yield chunk
                        except queue.Empty:
                            break

                # 4. Cleanup Input
                video_process.stdin.close()
                
                # Yield remaining output
                while True:
                    chunk = output_queue.get() # Blocking wait for remainder
                    if chunk is None:
                        break
                    yield chunk
                
                t.join()
                video_process.wait()
                    
            except Exception as e:
                print(f"Stream Generator Error: {e}")
                video_process.kill()

        return stream_generator
