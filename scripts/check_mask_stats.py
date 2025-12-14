import os
import cv2
import numpy as np

MASK_DIR = "data/flood_dataset/masks"

def analyze_masks():
    if not os.path.exists(MASK_DIR):
        print("❌ Folder not found!")
        return

    files = [f for f in os.listdir(MASK_DIR) if f.endswith(".png")]
    if not files:
        print("❌ No masks in folder!")
        return

    print(f"🔍 Total {len(files)} masks found. Analyzing 5 random samples...\n")
    
    # Select 5 random files to analyze
    sample_files = np.random.choice(files, min(5, len(files)), replace=False)

    for fname in sample_files:
        path = os.path.join(MASK_DIR, fname)
        # Read mask as is (with 0-1-2-3-4 values)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # Find unique values and counts
        unique, counts = np.unique(mask, return_counts=True)
        stats = dict(zip(unique, counts))
        
        print(f"📄 File: {fname}")
        print(f"   Values: {stats}")
        
        # Interpret
        if len(unique) == 1 and 0 in unique:
            print("   ⚠️  STATUS: Completely empty (Background Only)")
        else:
            print("   ✅  STATUS: FULL! (Buildings detected)")
            if 1 in stats: print(f"      - No Damage Building: {stats[1]} pixels")
            if 2 in stats: print(f"      - Minor Damage: {stats[2]} pixels")
            if 3 in stats: print(f"      - Major Damage: {stats[3]} pixels")
            if 4 in stats: print(f"      - Destroyed: {stats[4]} pixels")
        print("-" * 40)

if __name__ == "__main__":
    analyze_masks()