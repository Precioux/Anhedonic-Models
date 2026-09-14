import os
from huggingface_hub import snapshot_download

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
DESTINATION = "/mnt/mahdipou/models/internvl2-5-8b"

def download_model():
    print(f"Starting download for: {MODEL_ID}")
    print(f"Destination: {DESTINATION}")
    
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=DESTINATION,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("Download status: SUCCESS")
    except Exception as e:
        print(f"Download status: FAILED")
        print(f"Error: {e}")

if __name__ == "__main__":
    download_model()