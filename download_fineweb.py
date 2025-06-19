#!/usr/bin/env python3
"""Download FineWeb-Edu 10BT sample dataset"""

from datasets import load_dataset
from datasets.config import HF_DATASETS_CACHE
import sys
import os

# Get the actual cache directory that will be used
cache_dir = os.path.expanduser(HF_DATASETS_CACHE)

print("Starting download of FineWeb-Edu 10BT sample...")
print(f"HF_HOME from environment: {os.environ.get('HF_HOME', 'Not set')}")
print(f"HF_DATASETS_CACHE from environment: {os.environ.get('HF_DATASETS_CACHE', 'Not set')}")
print(f"This will download to: {cache_dir}")
print("Expected size: ~10-20GB compressed")
print("\nPress Ctrl+C to cancel\n")

try:
    # Download the 10BT sample split
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT",
        split="train",
        streaming=True  # This will download and cache
    )
    
    print(f"\nDownload complete!")
    print(f"Dataset info:")
    print(f"- Features: {dataset.features}")
    print(f"- First example keys: {list(dataset[0].keys())}")
    
except KeyboardInterrupt:
    print("\nDownload cancelled by user")
    sys.exit(1)
except Exception as e:
    print(f"\nError downloading dataset: {e}")
    sys.exit(1)