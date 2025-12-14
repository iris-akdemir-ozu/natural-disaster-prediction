import os
import shutil
from pathlib import Path
import random

# ================= SETTINGS =================
# Path to the downloaded xView2 directory (Update this according to your path!)
SOURCE_DIR = "/Users/iris/Downloads/train"  # <-- Write the full path to the downloaded folder here
SOURCE_IMAGES = os.path.join(SOURCE_DIR, "images")
SOURCE_TARGETS = os.path.join(SOURCE_DIR, "targets")

# Target Directories (Your project's data path)
DEST_IMAGES = "data/flood_dataset/images"
DEST_MASKS = "data/flood_dataset/masks"
# ===========================================

def setup_dataset():
    # 1. Clean/Create Directories
    print(f"Preparing target directories: {DEST_IMAGES}")
    if os.path.exists(DEST_IMAGES): shutil.rmtree("data/flood_dataset")
    os.makedirs(DEST_IMAGES, exist_ok=True)
    os.makedirs(DEST_MASKS, exist_ok=True)

    # 2. Match and Copy Files
    print("Scanning files...")
    
    # We will take only 'post_disaster' images
    # Because damage assessment is done post-disaster.
    all_files = os.listdir(SOURCE_IMAGES)
    post_images = [f for f in all_files if "post" in f and f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Total {len(post_images)} post-disaster images found.")
    print("Copying process starting (This might take a while)...")

    count = 0
    for img_name in post_images:
        # Image path
        src_img_path = os.path.join(SOURCE_IMAGES, img_name)
        
        # Find target mask name
        # Usually names are the same in xView2 or have '_target' suffix.
        # First, check if a mask with the exact same name exists:
        src_mask_path = os.path.join(SOURCE_TARGETS, img_name)
        
        # If not, try common naming differences (e.g., _target.png instead of .png)
        if not os.path.exists(src_mask_path):
            name_part = os.path.splitext(img_name)[0]
            src_mask_path = os.path.join(SOURCE_TARGETS, name_part + "_target.png")
        
        # If mask is found, copy it
        if os.path.exists(src_mask_path):
            # Copy files to the new location
            shutil.copy2(src_img_path, os.path.join(DEST_IMAGES, img_name))
            shutil.copy2(src_mask_path, os.path.join(DEST_MASKS, img_name)) # Renaming mask to match the image name
            count += 1
            
            if count % 100 == 0:
                print(f"   {count} data processed...")
        else:
            # Skip image if no mask found
            continue

    print("="*50)
    print(f"PROCESS COMPLETED!")
    print(f"Total {count} images and masks matched and transferred.")
    print(f"Save Location: data/flood_dataset/")
    print("="*50)
    print("Now you can run: python scripts/train_unet_model.py")

if __name__ == "__main__":
    # Check if downloaded folder exists
    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Downloaded folder not found: {SOURCE_DIR}")
        print("Please update the 'SOURCE_DIR' line in the script with the path to the downloaded folder.")
    else:
        setup_dataset()