"""
Create a small synthetic dataset for testing the flood segmentation system
This generates random images and masks just to test the pipeline
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random

def create_synthetic_flood_data(num_samples=50, output_dir="data/flood_dataset"):
    """
    Generate synthetic flood images and masks for testing
    """
    print("="*60)
    print("Creating Synthetic Test Dataset")
    print("="*60)
    
    # Create directories
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    img_size = (256, 256)
    
    for i in range(num_samples):
        # Create synthetic satellite-like image
        # Generate base terrain colors (green, brown)
        img = Image.new('RGB', img_size)
        pixels = np.random.randint(50, 200, (img_size[1], img_size[0], 3), dtype=np.uint8)
        # Make it more greenish/brownish (terrain-like)
        pixels[:, :, 1] = np.clip(pixels[:, :, 1] + 30, 0, 255)  # More green
        img = Image.fromarray(pixels)
        
        # Create mask with random flood zones (white=flood, black=no flood)
        mask = Image.new('L', img_size, 0)  # Start with all black
        draw_mask = ImageDraw.Draw(mask)
        
        # Add 2-5 random flood zones (irregular shapes)
        num_zones = random.randint(2, 5)
        for _ in range(num_zones):
            # Random flood zone
            x = random.randint(0, img_size[0] - 50)
            y = random.randint(0, img_size[1] - 50)
            width = random.randint(30, 100)
            height = random.randint(30, 100)
            
            # Draw irregular ellipse for flood zone
            draw_mask.ellipse([x, y, x + width, y + height], fill=255)
        
        # Overlay blue tint on image where mask indicates flood
        img_array = np.array(img)
        mask_array = np.array(mask)
        
        # Make flooded areas more blue
        flood_mask_3d = mask_array[:, :, np.newaxis] / 255.0
        blue_overlay = np.array([100, 150, 255])
        img_array = img_array * (1 - flood_mask_3d * 0.6) + blue_overlay * flood_mask_3d * 0.6
        img = Image.fromarray(img_array.astype(np.uint8))
        
        # Save files
        img_filename = f"flood_sample_{i:03d}.png"
        mask_filename = f"flood_sample_{i:03d}.png"
        
        img.save(os.path.join(images_dir, img_filename))
        mask.save(os.path.join(masks_dir, mask_filename))
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_samples} samples...")
    
    print(f"\n✅ Successfully created {num_samples} synthetic samples!")
    print(f"📂 Images: {images_dir}")
    print(f"📂 Masks: {masks_dir}")
    print("\n⚠️  Note: This is synthetic data for TESTING ONLY")
    print("For real flood prediction, download actual satellite/aerial imagery datasets")
    print("\nYou can now run: python scripts/train_unet_model.py")


if __name__ == "__main__":
    create_synthetic_flood_data(num_samples=50)
