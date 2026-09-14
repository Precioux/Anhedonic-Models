import os
from huggingface_hub import snapshot_download, login

# Configuration
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DESTINATION = "/mnt/mahdipou/models/Mistral-7B-Instruct-v0.3"
MY_TOKEN = ""  # Replace with your actual Hugging Face token <---------------

def download_model():
    print(f"Starting download for: {MODEL_ID}")
    print(f"Destination: {DESTINATION}")
    
    try:
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