import os
import json
import numpy as np
import cv2
import shutil

# ================= SETTINGS =================
# Path to your xView2 Dataset Folder (Update this path with your own!)
# It should contain 'images' and 'labels' folders.
SOURCE_DIR = "/Users/iris/Downloads/train" 
IMAGES_DIR = os.path.join(SOURCE_DIR, "images")
LABELS_DIR = os.path.join(SOURCE_DIR, "labels")

# Target Folders (Inside your project)
OUTPUT_IMAGES = "data/flood_dataset/images"
OUTPUT_MASKS = "data/flood_dataset/masks"

# Color Codes (Class numbers for the model)
DAMAGE_MAP = {
    "no-damage": 1,       # 1: No Damage (Model learns this)
    "minor-damage": 2,    # 2: Minor Damage
    "major-damage": 3,    # 3: Major Damage
    "destroyed": 4,       # 4: Destroyed
    "un-classified": 1    # Treat unclassified as no damage
}
# ===========================================

def parse_wkt_polygon(wkt_str):
    """
    Converts a simple WKT format polygon into a numpy array.
    Ex: "POLYGON ((10 10, 20 20, 30 30))" -> [[10,10], [20,20], [30,30]]
    """
    try:
        # Clean unnecessary characters
        content = wkt_str.replace("POLYGON ((", "").replace("))", "")
        points = []
        for pair in content.split(", "):
            x, y = map(float, pair.split(" "))
            points.append([int(x), int(y)])
        return np.array([points], dtype=np.int32)
    except:
        return None

def create_masks():
    # 1. Clean and Prepare Directories
    if os.path.exists(OUTPUT_IMAGES): shutil.rmtree("data/flood_dataset")
    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    os.makedirs(OUTPUT_MASKS, exist_ok=True)

    print(f"🚀 Process Starting: {LABELS_DIR}")
    
    # Get only 'post_disaster' (post-event) files
    json_files = [f for f in os.listdir(LABELS_DIR) if f.endswith(".json") and "post_disaster" in f]
    print(f"📄 JSON Files Found: {len(json_files)}")

    processed_count = 0

    for json_file in json_files:
        # File paths
        json_path = os.path.join(LABELS_DIR, json_file)
        img_name = json_file.replace(".json", ".png") # xView2 usually uses .png
        src_img_path = os.path.join(IMAGES_DIR, img_name)

        if not os.path.exists(src_img_path):
            continue

        # Read JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Get Original Image Dimensions (Usually 1024x1024)
        img = cv2.imread(src_img_path)
        if img is None: continue
        h, w = img.shape[:2]

        # Create Empty Black Mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Iterate through each building in 'xy'
        # (In xView2 coordinates are located under 'xy')
        features = data.get('features', {}).get('xy', [])
        
        has_labels = False
        for feature in features:
            # Get damage type (no-damage, destroyed etc.)
            subtype = feature.get('properties', {}).get('subtype', 'no-damage')
            color_val = DAMAGE_MAP.get(subtype, 1)

            # Get Coordinates (WKT) and draw
            wkt = feature.get('wkt', '')
            if wkt:
                poly_points = parse_wkt_polygon(wkt)
                if poly_points is not None:
                    # Draw filled polygon on mask
                    cv2.fillPoly(mask, poly_points, color_val)
                    has_labels = True

        # Save Files
        # Copy Image
        shutil.copy2(src_img_path, os.path.join(OUTPUT_IMAGES, img_name))
        
        # Save Mask (As uncompressed PNG)
        cv2.imwrite(os.path.join(OUTPUT_MASKS, img_name), mask)

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"   ✅ {processed_count} images processed...")

    print("="*50)
    print(f"🎉 COMPLETED! Total {processed_count} training samples prepared.")
    print("👉 Now run this command: python scripts/train_unet_model.py")

if __name__ == "__main__":
    create_masks()