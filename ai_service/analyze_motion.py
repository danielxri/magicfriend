
import cv2
import numpy as np
import sys

def analyze_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    if len(frames) < 2:
        print("Video too short")
        return

    # Assume mouth is center-ish or just check global variance difference
    # Or just check if frames are identical
    
    diffs = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype("float") - frames[i-1].astype("float"))
        diffs.append(diff.mean())
    
    print(f"Motion Frame-to-Frame Diff Mean: {np.mean(diffs):.4f}")
    if np.mean(diffs) < 0.1:
        print("STATIC VIDEO DETECTED")
    else:
        print("MOTION DETECTED")

if __name__ == "__main__":
    analyze_motion(sys.argv[1])
