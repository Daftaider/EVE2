#!/usr/bin/env python3
"""
Download script for EVE2 models.

This script downloads the necessary ML models for the EVE2 system:
- Whisper model for speech recognition
- LLaMA model for language processing
- Piper TTS model for speech synthesis
"""

import os
import sys
import argparse
import hashlib
import logging
import requests
import shutil
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eve.config import config, MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("model_downloader")

# Model information
MODEL_INFO = {
    "whisper": {
        "name": "whisper-small.en.gguf",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin?download=true",
        "size": 466_000_000,
        "md5": None,  # Skip MD5 check
        "dest_path": MODELS_DIR / "whisper-small.en.gguf",
    },
    "llama": {
        "name": "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",  # Use a smaller model that's public
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf?download=true",
        "size": 874_000_000,
        "md5": None,  # Skip MD5 check
        "dest_path": MODELS_DIR / "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
    },
    "tts": {
        "name": "tts-piper-en",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true",
        "size": 63_200_000,
        "md5": None,  # Skip MD5 check
        "dest_path": MODELS_DIR / "tts-piper-en/en_US/lessac_medium.onnx",
        "is_archive": False,  # Changed from archive to direct file
    },
    "tts_config": {
        "name": "tts-piper-en-config",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true",
        "size": 5_000,
        "md5": None,  # Skip MD5 check
        "dest_path": MODELS_DIR / "tts-piper-en/en_US/lessac_medium.onnx.json",
    },
    "emotion": {
        "name": "emotion-fer",
        "url": "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5",
        "size": 4_000_000,
        "md5": None,  # Skip MD5 check
        "dest_path": MODELS_DIR / "emotion-fer.h5",
    },
}


def check_md5(file_path: Path, expected_md5: str) -> bool:
    """
    Check if a file's MD5 hash matches the expected value.
    
    Args:
        file_path: Path to the file.
        expected_md5: Expected MD5 hash.
        
    Returns:
        True if the MD5 hash matches, False otherwise.
    """
    if not file_path.exists():
        return False
    
    # Calculate MD5 hash
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest() == expected_md5


def download_file(url: str, dest_path: Path, expected_size: int, desc: str = "Downloading") -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from.
        dest_path: Path to save the file.
        expected_size: Expected file size in bytes.
        desc: Description for the progress bar.
        
    Returns:
        True if successful, False otherwise.
    """
    # Create parent directory if it doesn't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download with progress bar
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", expected_size))
        
        with open(dest_path, "wb") as f, tqdm(
            desc=desc,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """
    Extract an archive file.
    
    Args:
        archive_path: Path to the archive file.
        dest_dir: Directory to extract to.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                for member in tqdm(zip_ref.infolist(), desc="Extracting"):
                    zip_ref.extract(member, dest_dir)
                
        elif archive_path.suffix == ".gz" and archive_path.suffixes[-2:] == [".tar", ".gz"]:
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                for member in tqdm(tar_ref.getmembers(), desc="Extracting"):
                    tar_ref.extract(member, dest_dir)
        
        else:
            logger.error(f"Unsupported archive format: {archive_path}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error extracting {archive_path}: {e}")
        return False


def download_model(model_type: str, force: bool = False) -> bool:
    """
    Download a specific model.
    
    Args:
        model_type: Type of model to download.
        force: Force download even if the model already exists.
        
    Returns:
        True if successful, False otherwise.
    """
    if model_type not in MODEL_INFO:
        logger.error(f"Unknown model type: {model_type}")
        return False
    
    info = MODEL_INFO[model_type]
    dest_path = info["dest_path"]
    md5 = info.get("md5", "")
    
    # Check if model already exists and has the correct MD5
    if not force and dest_path.exists():
        if not md5 or check_md5(dest_path, md5):
            logger.info(f"Model {info['name']} already exists")
            return True
        else:
            logger.warning(f"Model {info['name']} exists but has incorrect MD5, re-downloading")
    
    # Download the model
    logger.info(f"Downloading {info['name']} from {info['url']}")
    
    if info.get("is_archive", False):
        # For archives, download to a temporary file and then extract
        temp_path = dest_path.with_suffix(".tmp")
        
        if download_file(info["url"], temp_path, info["size"], desc=f"Downloading {info['name']}"):
            # Extract the archive
            if extract_archive(temp_path, dest_path):
                # Delete the temporary file
                temp_path.unlink()
                logger.info(f"Successfully downloaded and extracted {info['name']}")
                return True
            else:
                logger.error(f"Failed to extract {info['name']}")
                return False
        else:
            logger.error(f"Failed to download {info['name']}")
            return False
    
    else:
        # For regular files, download directly to the destination
        if download_file(info["url"], dest_path, info["size"], desc=f"Downloading {info['name']}"):
            logger.info(f"Successfully downloaded {info['name']}")
            return True
        else:
            logger.error(f"Failed to download {info['name']}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download models for EVE2")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_INFO.keys()) + ["all"],
        default=["all"],
        help="Models to download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if models already exist",
    )
    args = parser.parse_args()
    
    models_to_download = list(MODEL_INFO.keys()) if "all" in args.models else args.models
    
    logger.info(f"Will download {len(models_to_download)} models")
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download each model
    success = True
    for model in models_to_download:
        logger.info(f"Processing {model} model")
        if not download_model(model, args.force):
            success = False
    
    if success:
        logger.info("All models downloaded successfully")
    else:
        logger.warning("Some models failed to download")
        sys.exit(1)


if __name__ == "__main__":
    main() 