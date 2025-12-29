import os
import glob
import shutil
import re
from pathlib import Path

# Paths
MUSE_DIR = Path("ai_service/MuseTalk")
DISTILL_OUTPUT = MUSE_DIR / "distill_output"
PROD_MODEL_PATH = MUSE_DIR / "models/musetalk/student_prod.pth"
APP_FILE = Path("ai_service/app_student.py")

def main():
    print(f"Scanning for checkpoints in {DISTILL_OUTPUT}...")
    
    # Find all student_step_*.pth files
    checkpoints = list(DISTILL_OUTPUT.glob("student_step_*.pth"))
    
    if not checkpoints:
        print("Error: No checkpoints found!")
        return

    # Sort by step number
    # extract number from student_step_123.pth
    def get_step(p):
        match = re.search(r"student_step_(\d+).pth", p.name)
        return int(match.group(1)) if match else -1

    # Sort by modification time (newest first) to handle counter resets
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    latest_step = get_step(latest_ckpt)
    
    print(f"Found latest checkpoint: {latest_ckpt.name} (Step {latest_step})")
    
    # Validation: Ensure it's not a tiny empty file
    if latest_ckpt.stat().st_size < 100_000_000: # < 100MB
        print("Warning: Checkpoint seems too small. Aborting deployment.")
        return

    # Deploy
    print(f"Deploying to {PROD_MODEL_PATH}...")
    PROD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest_ckpt, PROD_MODEL_PATH)
    print("Deployment successful.")
    
    # Update app_student.py if needed
    update_app_config(latest_step)

def update_app_config(step):
    print("Updating app_student.py to point to production model...")
    with open(APP_FILE, 'r') as f:
        content = f.read()

    # Pattern to find the STUDENT_CHECKPOINT line
    # STUDENT_CHECKPOINT = MUSE_DIR / "distill_output/student_step_3000.pth"
    # Replace with: STUDENT_CHECKPOINT = MUSE_DIR / "models/musetalk/student_prod.pth"
    
    new_line = 'STUDENT_CHECKPOINT = MUSE_DIR / "models/musetalk/student_prod.pth"'
    
    # Regex to match the assignment
    pattern = r'STUDENT_CHECKPOINT\s*=\s*.*'
    
    if "student_prod.pth" in content:
        print("app_student.py is already configured for production model.")
    else:
        new_content = re.sub(pattern, new_line, content)
        with open(APP_FILE, 'w') as f:
            f.write(new_content)
        print("app_student.py updated.")

if __name__ == "__main__":
    main()
