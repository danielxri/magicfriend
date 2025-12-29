import os
import glob
import subprocess
import cv2

video_root = "/app/ai_service/MuseTalk/dataset/HDTF/video_audio_clip_root"

def get_duration(vid_path):
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0: return 0
    return frames / fps

def generate_audio():
    videos = glob.glob(os.path.join(video_root, "*.mp4"))
    for vid in videos:
        # We need both .wav (for initial check possible?) and _16k.wav (for ensure_wav)
        # dataset.py calls ensure_wav(vid_path), which looks for vid_path + "_16k.wav"
        
        base, ext = os.path.splitext(vid)
        wav_path_16k = base + "_16k.wav"
        
        if os.path.exists(wav_path_16k):
            print(f"Skipping {wav_path_16k} (exists)")
            continue
            
        duration = get_duration(vid)
        if duration <= 0:
            print(f"Skipping {vid} (invalid duration)")
            continue
            
        print(f"Generating audio for {vid} -> {wav_path_16k} ({duration:.2f}s)")
        
        # Generate 16kHz sine wave
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-ar", "16000", "-ac", "1",
            wav_path_16k
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    generate_audio()
