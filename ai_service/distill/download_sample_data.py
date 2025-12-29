import os
import requests
from tqdm import tqdm

# Official MuseTalk Demo Videos (from README/GitHub)
VIDEO_URLS = {
    "sun_demo.mp4": "https://github.com/TMElyralab/MuseTalk/assets/163980830/37a3a666-7b90-4244-8d3a-058cb0e44107",
    "musk_demo.mp4": "https://github.com/TMElyralab/MuseTalk/assets/163980830/4a4bb2d1-9d14-4ca9-85c8-7f19c39f712e",
    "yongen_demo.mp4": "https://github.com/TMElyralab/MuseTalk/assets/163980830/57ef9dee-a9fd-4dc8-839b-3fbbbf0ff3f4",
    "sit_demo.mp4": "https://github.com/TMElyralab/MuseTalk/assets/163980830/5fbab81b-d3f2-4c75-abb5-14c76e51769e"
}

TARGET_DIR = "/app/ai_service/MuseTalk/dataset/HDTF/source"
LOCAL_TARGET_DIR = "/app/ai_service/MuseTalk/dataset/HDTF/source"

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    save_path = os.path.join(LOCAL_TARGET_DIR, filename)
    
    print(f"Downloading {filename}...")
    with open(save_path, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    # Ensure directory exists
    if not os.path.exists(LOCAL_TARGET_DIR):
        print(f"Creating directory: {LOCAL_TARGET_DIR}")
        os.makedirs(LOCAL_TARGET_DIR, exist_ok=True)
    
    for name, url in VIDEO_URLS.items():
        try:
            download_file(url, name)
            # Verify it's a valid video file (simple size check)
            if os.path.getsize(os.path.join(LOCAL_TARGET_DIR, name)) < 1000:
                print(f"WARNING: {name} seems too small. Might be a broken link.")
        except Exception as e:
            print(f"Failed to download {name}: {e}")

    print("\nSample dataset download complete.")
    print(f"Videos saved to: {LOCAL_TARGET_DIR}")

if __name__ == "__main__":
    main()
