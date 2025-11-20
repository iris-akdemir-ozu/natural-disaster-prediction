# Complete Usage Guide

Step-by-step guide to use the Flood Segmentation System from start to finish.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Detailed Workflow](#detailed-workflow)
3. [Dataset Preparation](#dataset-preparation)
4. [Model Training](#model-training)
5. [Using the Web Interface](#using-the-web-interface)
6. [API Usage](#api-usage)
7. [Advanced Features](#advanced-features)
8. [Tips and Best Practices](#tips-and-best-practices)

## Quick Start

### For First-Time Users

\`\`\`bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download sample dataset
python scripts/download_sample_dataset.py
# Follow instructions to get dataset

# 3. Train model
python scripts/train_unet_model.py

# 4. Start backend
cd backend
python app.py

# 5. Start frontend (in new terminal)
npm install
npm run dev

# 6. Open browser
# Visit: http://localhost:5173
\`\`\`

## Detailed Workflow

### Complete Project Workflow

\`\`\`
1. Data Collection → 2. Data Preparation → 3. Model Training
                                                ↓
                                          4. Model Evaluation
                                                ↓
                                         5. Backend Setup
                                                ↓
                                         6. Web Interface
                                                ↓
                                         7. Make Predictions
\`\`\`

## Dataset Preparation

### Option 1: Kaggle Dataset

1. Visit [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Search for "flood segmentation" or "water segmentation"
3. Download dataset (e.g., "Flood Area Segmentation")
4. Extract to \`data/flood_dataset/\`
5. Organize into structure:
   \`\`\`
   data/flood_dataset/
   ├── images/     # Satellite/aerial images
   │   ├── img_001.png
   │   ├── img_002.png
   │   └── ...
   └── masks/      # Binary flood masks
       ├── mask_001.png
       ├── mask_002.png
       └── ...
   \`\`\`

### Option 2: Sen1Floods11 (Recommended for Production)

1. Search "Sen1Floods11 dataset" on Google
2. Download from official source or Hugging Face
3. Contains 4,831 high-quality images
4. Includes both optical and radar imagery
5. Pre-labeled flood masks

### Option 3: Custom Dataset

If creating your own dataset:

1. **Images**: Satellite or aerial imagery of flood-prone areas
2. **Masks**: Binary masks where:
   - White (255) = Flood area
   - Black (0) = No flood
3. **Requirements**:
   - Same filename for image and corresponding mask
   - PNG or JPG format
   - Minimum 100 images (500+ recommended)
   - Square images preferred (256x256 or 512x512)

## Model Training

### Basic Training

\`\`\`bash
python scripts/train_unet_model.py
\`\`\`

### Training Configuration

Edit \`scripts/train_unet_model.py\` to adjust:

\`\`\`python
IMG_HEIGHT = 256       # Image dimensions
IMG_WIDTH = 256
BATCH_SIZE = 16        # Reduce if out of memory
EPOCHS = 50            # Training iterations
LEARNING_RATE = 1e-4   # Learning rate
\`\`\`

### Understanding Training Output

During training, you'll see:

\`\`\`
Epoch 1/50
10/10 [==============================] - 45s 4s/step
  loss: 0.3456
  dice_coefficient: 0.7123
  iou_metric: 0.6234
  val_loss: 0.3789
  val_dice_coefficient: 0.6891
\`\`\`

**Key Metrics:**
- **Loss**: Lower is better (< 0.3 is good)
- **Dice Coefficient**: Higher is better (> 0.75 is excellent)
- **IoU**: Higher is better (> 0.65 is excellent)

### Training Tips

1. **Start Small**: Train on 100-200 images first to verify everything works
2. **Monitor Validation**: If validation metrics stop improving, stop training
3. **Save Checkpoints**: Best model is automatically saved
4. **GPU Recommended**: Training is much faster with GPU

## Using the Web Interface

### Step 1: Start Services

Terminal 1 - Backend:
\`\`\`bash
cd backend
python app.py
\`\`\`

Terminal 2 - Frontend:
\`\`\`bash
npm run dev
\`\`\`

### Step 2: Access Interface

Open browser: \`http://localhost:5173\`

### Step 3: Upload Image

1. Click "Click to upload image" button
2. Select satellite/aerial image (PNG/JPG)
3. Image preview appears

### Step 4: Analyze

1. Click "Analyze Image" button
2. Wait for processing (5-10 seconds)
3. Results appear in Results Panel

### Step 5: View Results

**Results Panel Shows:**
- Risk Level (Low/Medium/High/Critical)
- Flood Coverage percentage
- Confidence scores

**Map Viewer Shows:**
- Original image overlay
- Flood zones highlighted in red
- Interactive map with zoom/pan

## API Usage

### Direct API Calls

#### Single Prediction

\`\`\`bash
curl -X POST \
  http://localhost:5000/api/predict \
  -F "image=@path/to/image.jpg"
\`\`\`

Response:
\`\`\`json
{
  "success": true,
  "prediction": {
    "flood_detected": true,
    "flood_percentage": 23.5,
    "max_confidence": 0.95,
    "avg_confidence": 0.78,
    "risk_level": "High"
  },
  "masks": {
    "probability_mask": "data:image/png;base64,...",
    "binary_mask": "data:image/png;base64,...",
    "overlay": "data:image/png;base64,..."
  }
}
\`\`\`

#### Batch Prediction

\`\`\`bash
curl -X POST \
  http://localhost:5000/api/batch-predict \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg"
\`\`\`

### Python API Client

\`\`\`python
import requests

def predict_flood(image_path):
    url = "http://localhost:5000/api/predict"
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

# Use it
result = predict_flood("test_image.jpg")
print(f"Flood detected: {result['prediction']['flood_detected']}")
print(f"Risk level: {result['prediction']['risk_level']}")
\`\`\`

### JavaScript API Client

\`\`\`javascript
async function predictFlood(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch('http://localhost:5000/api/predict', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}

// Use it
const file = document.querySelector('input[type="file"]').files[0];
const result = await predictFlood(file);
console.log('Risk level:', result.prediction.risk_level);
\`\`\`

## Advanced Features

### Standalone Inference

Test model without web interface:

\`\`\`bash
# Single image
python scripts/inference_test.py test_image.jpg

# Batch processing
python scripts/inference_test.py test_images_folder/
\`\`\`

### Coordinate-Based Predictions

For geographic mapping:

\`\`\`bash
curl -X POST \
  http://localhost:5000/api/predict-coordinates \
  -F "image=@map.jpg" \
  -F "bounds=40.7128,-74.0060,40.7828,-73.9260"
\`\`\`

Returns GeoJSON format for Leaflet integration.

## Tips and Best Practices

### For Best Predictions

1. **Image Quality**: Use high-resolution satellite imagery
2. **Consistent Scale**: Images at similar zoom levels work better
3. **Clear Weather**: Cloud-free images produce better results
4. **Recent Data**: Use recent imagery for current conditions

### Improving Model Accuracy

1. **More Data**: Increase training dataset size
2. **Better Labels**: Ensure masks accurately represent floods
3. **Data Augmentation**: Already implemented (flips, rotations)
4. **Longer Training**: Increase epochs if model still improving
5. **Fine-tuning**: Train on region-specific data

### Performance Optimization

1. **Reduce Image Size**: Smaller images process faster
2. **Batch Processing**: Use batch API for multiple images
3. **GPU Usage**: Use GPU-enabled server for production
4. **Caching**: Implement caching for repeated predictions

### Common Issues

**Issue**: Model predicts all flood or no flood
- **Solution**: Check mask quality, increase dataset size

**Issue**: Out of memory during training
- **Solution**: Reduce BATCH_SIZE or IMG_HEIGHT/IMG_WIDTH

**Issue**: Predictions take too long
- **Solution**: Use GPU, reduce image size, or optimize model

**Issue**: CORS errors in web interface
- **Solution**: Ensure backend CORS is enabled (already done)

## Next Steps

1. **Improve Model**: Collect more data and retrain
2. **Deploy to Production**: Follow DEPLOYMENT.md guide
3. **Add Features**: Implement time-series prediction, alerts
4. **Scale Up**: Add load balancing, caching, monitoring

## Support

For issues or questions:
1. Check troubleshooting section in README.md
2. Review training logs for model issues
3. Check browser console for frontend errors
4. Check Flask logs for backend errors
\`\`\`
