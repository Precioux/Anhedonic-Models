import os
from huggingface_hub import snapshot_download

# The exact model ID on Hugging Face
model_id = "Qwen/Qwen3-VL-8B-Instruct"

# IMPORTANT: Change this path to your scratch directory on the RCP cluster. !!!!!!!!!!!
save_dir = "/mnt/mahdipou/models/Qwen3-VL-8B-Instruct" 

print(f"🚀 Downloading model '{model_id}'...")
print(f"📂 Save directory: {save_dir}")

# Download all model files
snapshot_download(
    repo_id=model_id,
    local_dir=save_dir,
    local_dir_use_symlinks=False, # Disable symlinks for a hard copy (safer on clusters)
    resume_download=True,         # Resume if interrupted
    max_workers=4                 # Parallel downloading for faster speed
)

print("✅ Download completed successfully! You can now load the model.")