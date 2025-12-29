import os
import argparse
import subprocess
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from typing import Tuple, List, Union
import json
import cv2
from musetalk.utils.face_detection import FaceAlignment,LandmarksType
import sys
import glob

# Reuse existing classes/funcs from preprocess.py via import or copy?
# Copying key parts to ensure standalone behavior and correct patching (FaceAlignment)

class AnalyzeFace:
    def __init__(self, device: Union[str, torch.device], config_file: str, checkpoint_file: str):
        self.device = device
        self.facedet = FaceAlignment(LandmarksType._2D, flip_input=False, device=self.device)

    def __call__(self, im: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        try:
            if im.ndim == 3:
                im = np.expand_dims(im, axis=0)
            elif im.ndim != 4 or im.shape[0] != 1:
                raise ValueError("Input image must have shape (1, H, W, C)")
            
            bbox_batch = self.facedet.get_detections_for_batch(np.asarray(im))
            
            landmark_results = []
            if bbox_batch and len(bbox_batch) > 0 and bbox_batch[0] is not None:
                landmark_results = self.facedet.get_landmarks(np.asarray(im)[0])
            
            if landmark_results and len(landmark_results) > 0:
                 face_land_mark = landmark_results[0]
                 face_land_mark = np.array(face_land_mark).astype(np.int32)
            else:
                 face_land_mark = np.array([])

            return face_land_mark, bbox_batch
        
        except Exception as e:
            print(f"Error during face analysis: {e}")
            return np.array([]),[] 

def convert_video(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    for idx, vid in enumerate(vid_list):
        if vid.endswith('.mp4'):
            org_vid_path = os.path.join(org_path, vid)
            dst_vid_path = os.path.join(dst_path, vid)
                
            if org_vid_path != dst_vid_path:
                # Use -c:a copy to preserve audio!
                cmd = [
                    "ffmpeg", "-hide_banner", "-y", "-i", org_vid_path, 
                    "-r", "25", "-crf", "15", "-c:v", "libx264", "-c:a", "copy",
                    "-pix_fmt", "yuv420p", dst_vid_path
                ]
                subprocess.run(cmd, check=True)

def segment_video(org_path: str, dst_path: str, vid_list: List[str], segment_duration: int = 30) -> None:
    for idx, vid in enumerate(vid_list):
        if vid.endswith('.mp4'):
            input_file = os.path.join(org_path, vid)
            original_filename = os.path.basename(input_file)

            command = [
                'ffmpeg', '-hide_banner', '-y', '-i', input_file, '-c', 'copy', '-map', '0',
                '-segment_time', str(segment_duration), '-f', 'segment',
                '-reset_timestamps', '1',
                os.path.join(dst_path, f'clip%03d_{original_filename}')
            ]
            subprocess.run(command, check=True)

def extract_audio(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    for idx, vid in enumerate(vid_list):
        if vid.endswith('.mp4'):
            video_path = os.path.join(org_path, vid)
            audio_output_path = os.path.join(dst_path, os.path.splitext(vid)[0] + ".wav")
            try:
                command = [
                    'ffmpeg', '-hide_banner', '-y', '-i', video_path,
                    '-vn', '-acodec', 'pcm_s16le', '-f', 'wav',
                    '-ar', '16000', '-ac', '1', audio_output_path,
                ]
                subprocess.run(command, check=True)
                print(f"Audio saved to: {audio_output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error extracting audio from {vid}: {e}")

def analyze_video(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # DWPose not needed since using FaceAlignment directly
    
    analyze_face = AnalyzeFace(device, None, None)
    
    for vid in tqdm(vid_list, desc="Processing videos"):
        if vid.endswith('.mp4'):
            vid_path = os.path.join(org_path, vid)
            wav_path = vid_path.replace(".mp4",".wav")
            vid_meta = os.path.join(dst_path, os.path.splitext(vid)[0] + ".json")
            
            print(f'process video {vid}')

            total_bbox_list = []
            total_pts_list = []
            isvalid = True

            video_height = video_width = face_height = face_width = 0

            try:
                # Use CV2
                cap = cv2.VideoCapture(vid_path)
                if not cap.isOpened():
                    print(f"Failed to open {vid_path}")
                    continue
            except Exception as e:
                print(e)
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Just verify first few frames logic? No, need full JSON for dataset loader.
            
            for frame_idx in range(total_frames):
                ret, frame_bgr = cap.read()
                if not ret: break
                    
                if frame_idx==0:
                    video_height,video_width,_ = frame_bgr.shape
                
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                pts_list, bbox_list = analyze_face(frame_rgb)
                
                if len(bbox_list)>0 and None not in bbox_list:
                    bbox = bbox_list[0]
                else:
                    isvalid = False
                    bbox = []
                    # print(f"Invalid frame {frame_idx}")
                    break

                if len(pts_list)>0:
                    pts = pts_list.tolist()
                else:
                    isvalid = False
                    pts = []
                    break

                if frame_idx==0:
                    x1,y1,x2,y2 = bbox 
                    face_height, face_width = y2-y1,x2-x1

                total_pts_list.append(pts)
                total_bbox_list.append(bbox)

            meta_data = {
                    "mp4_path": vid_path,
                     "wav_path": wav_path,
                     "video_size": [video_height, video_width],
                     "face_size": [face_height, face_width],
                     "frames": total_frames,
                     "face_list": total_bbox_list,
                     "landmark_list": total_pts_list,
                     "isvalid":isvalid,
            }        
            with open(vid_meta, 'w') as f:
                json.dump(meta_data, f, indent=4)

def main():
    config_path = "./configs/training/preprocess.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Override paths to subset folders? No, use same folders but filter list.
    
    os.makedirs(cfg.video_root_25fps, exist_ok=True)
    os.makedirs(cfg.video_audio_clip_root, exist_ok=True)
    os.makedirs(cfg.meta_root, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_clip_file_list_train), exist_ok=True)

    # Pick 5 videos
    raw_vids = sorted(os.listdir(cfg.video_root_raw))
    subset_vids = raw_vids[:3] # Process 3 videos
    print(f"Processing subset: {subset_vids}")
    
    # 1. Convert
    convert_video(cfg.video_root_raw, cfg.video_root_25fps, subset_vids)
    
    # 2. Segment
    segment_video(cfg.video_root_25fps, cfg.video_audio_clip_root, subset_vids, segment_duration=cfg.clip_len_second)
    
    # 3. Extract Audio (Find clips generated from subset)
    # Filter clip dir for subset names
    all_clips = os.listdir(cfg.video_audio_clip_root)
    subset_clips = [c for c in all_clips if any(orig in c for orig in subset_vids) and c.endswith('.mp4')]
    
    extract_audio(cfg.video_audio_clip_root, cfg.video_audio_clip_root, subset_clips)
    
    # 4. Analyze
    analyze_video(cfg.video_audio_clip_root, cfg.meta_root, subset_clips)
    
    # 5. Generate Train List
    # Only include processed clips
    processed_meta = [f for f in os.listdir(cfg.meta_root) if f.replace('.json', '.mp4') in subset_clips]
    # Actually meta uses clip name.
    
    train_subset_path = "./dataset/HDTF/train_subset.txt"
    with open(train_subset_path, 'w') as f:
        for json_file in processed_meta:
            clip_name = json_file.replace('.json', '')
            f.write(clip_name + '\n')
            
    print(f"Done. Subset train list saved to {train_subset_path}")

if __name__ == "__main__":
    main()
