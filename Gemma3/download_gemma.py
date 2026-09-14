import os
from huggingface_hub import snapshot_download, login

# Configuration
MODEL_ID = "google/gemma-3-4b-it"
DESTINATION = "/mnt/mahdipou/models/gemma-3-4b"
MY_TOKEN = ""  # Replace with your actual Hugging Face token <---------------

def download_model():
    print(f"Starting download for: {MODEL_ID}")
    print(f"Destination: {DESTINATION}")
    
    try:
        # Login to resolve gated repository access
        login(token=MY_TOKEN)
        
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=DESTINATION,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=MY_TOKEN
        )
        print("Download status: SUCCESS")
    except Exception as e:
        print(f"Download status: FAILED")
        print(f"Error: {e}")

if __name__ == "__main__":
    download_model()