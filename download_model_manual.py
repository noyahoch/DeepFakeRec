"""Manually download XLS-R model files using git-lfs or direct download.

This script helps download the model files manually when SSL/certificate issues occur.
"""
import os
from pathlib import Path

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
LOCAL_DIR = Path("models") / "wav2vec2-xls-r-300m"

# Files needed for the model
REQUIRED_FILES = [
    "config.json",
    "pytorch_model.bin",  # or model.safetensors
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]

print(f"Manual download instructions for {MODEL_NAME}")
print("=" * 60)
print(f"\n1. Create directory: {LOCAL_DIR}")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
print(f"   ✓ Created: {LOCAL_DIR.absolute()}")

print("\n2. Download files using one of these methods:")
print("\n   Method A: Git LFS (Recommended)")
print(f"   git lfs install")
print(f"   git clone https://huggingface.co/{MODEL_NAME} {LOCAL_DIR}")

print("\n   Method B: HuggingFace Hub CLI")
print(f"   pip install huggingface_hub")
print(f"   huggingface-cli download {MODEL_NAME} --local-dir {LOCAL_DIR}")

print("\n   Method C: Manual download from browser")
print(f"   Visit: https://huggingface.co/{MODEL_NAME}/tree/main")
print(f"   Download these files:")
for f in REQUIRED_FILES:
    print(f"     - {f}")
print(f"   Save them to: {LOCAL_DIR.absolute()}")

print("\n3. After downloading, update run_configs/config.yaml:")
print(f"   model:")
print(f"     pretrained_name: \"{LOCAL_DIR}\"  # Use relative or absolute path")

print("\n4. Verify the model loads:")
print(f"   uv run python -c \"from transformers import Wav2Vec2Model; m = Wav2Vec2Model.from_pretrained('{LOCAL_DIR}', local_files_only=True); print('✓ Model loaded!')\"")

print("\n" + "=" * 60)
