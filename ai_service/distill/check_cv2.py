import cv2
import os

path = "/app/ai_service/MuseTalk/dataset/HDTF/video_audio_clip_root/clip000_sun_demo.mp4"

def check():
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Failed to open video")
        return

    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Total frames: {frames}")
    
    ret, frame = cap.read()
    if ret:
        print(f"Read frame 0: {frame.shape}")
    else:
        print("Failed to read frame 0")

if __name__ == "__main__":
    check()
