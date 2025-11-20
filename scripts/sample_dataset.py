"""
Helper script to download sample flood dataset for quick testing
Downloads from Kaggle or other sources
"""

import os
import urllib.request
import zipfile
from pathlib import Path

def download_file(url, destination):
    """Download file with progress"""
    print(f"Downloading to {destination}...")
    
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)
        print(f"\rProgress: {percent:.1f}%", end='')
    
    urllib.request.urlretrieve(url, destination, progress)
    print("\nDownload complete!")


def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")


def setup_sample_dataset():
    """
    Download and setup a sample flood dataset
    
    Note: This is a placeholder. You need to:
    1. Find a suitable dataset (Kaggle, Roboflow, etc.)
    2. Get the direct download link
    3. Update this script with the correct URL
    """
    
    print("="*60)
    print("Sample Dataset Setup")
    print("="*60)
    print("\nThis script helps you download a sample dataset.")
    print("\nRecommended datasets:")
    print("1. Kaggle: 'Flood Area Segmentation'")
    print("2. Sen1Floods11 (Professional dataset)")
    print("3. Roboflow Universe flood datasets")
    
    print("\n" + "="*60)
    print("Manual Setup Instructions:")
    print("="*60)
    
    print("\nFor Kaggle datasets:")
    print("  1. Go to kaggle.com/datasets")
    print("  2. Search for 'flood segmentation' or 'water segmentation'")
    print("  3. Download the dataset")
    print("  4. Extract to: data/flood_dataset/")
    print("  5. Ensure structure:")
    print("     data/flood_dataset/")
    print("       ├── images/")
    print("       └── masks/")
    
    print("\nFor Sen1Floods11:")
    print("  1. Search 'Sen1Floods11 dataset'")
    print("  2. Download from the official source or Hugging Face")
    print("  3. Extract and organize into images/ and masks/")
    
    print("\nFor Roboflow Universe:")
    print("  1. Go to universe.roboflow.com")
    print("  2. Search 'flood' or 'water'")
    print("  3. Filter by 'Semantic Segmentation'")
    print("  4. Download in your preferred format")
    print("  5. Extract to data/flood_dataset/")
    
    print("\n" + "="*60)
    
    # Create directories
    os.makedirs("data/flood_dataset/images", exist_ok=True)
    os.makedirs("data/flood_dataset/masks", exist_ok=True)
    
    print("\nDirectories created:")
    print("  ✓ data/flood_dataset/images/")
    print("  ✓ data/flood_dataset/masks/")
    
    print("\nReady to add your dataset!")


if __name__ == "__main__":
    setup_sample_dataset()
