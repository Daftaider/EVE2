"""
Script to download the TinyLlama model.
"""
import os
import logging
import requests
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

def download_file(url: str, destination: str) -> bool:
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
        logger.error(f"Error downloading file: {e}")
        return False

def main():
    """Download TinyLlama model."""
    # Create model directory
    model_dir = Path("src/models/llm")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # TinyLlama model URL
    model_url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    model_path = model_dir / "tinyllama.gguf"
    
    if model_path.exists():
        logger.info("Model already exists, skipping download")
        return
        
    logger.info("Downloading TinyLlama model...")
    if download_file(model_url, str(model_path)):
        logger.info("Model downloaded successfully")
    else:
        logger.error("Failed to download model")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 