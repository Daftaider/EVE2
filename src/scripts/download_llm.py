"""
Script to download the TinyLlama model.
"""
import os
import requests
from tqdm import tqdm
from pathlib import Path

MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_DIR = Path("src/models/llm")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

def download_file(url: str, destination: Path) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for data in response.iter_content(block_size):
                    pbar.update(len(data))
                    f.write(data)
                    
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def main():
    """Main function."""
    # Create model directory if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    if MODEL_PATH.exists():
        print(f"Model already exists at {MODEL_PATH}")
        return
        
    print(f"Downloading model to {MODEL_PATH}")
    if download_file(MODEL_URL, MODEL_PATH):
        print("Model downloaded successfully")
    else:
        print("Failed to download model")

if __name__ == "__main__":
    main() 