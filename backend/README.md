# Flask Backend API

RESTful API for flood segmentation inference using the trained U-Net model.

## Features

- Single image prediction with confidence scores
- Batch prediction for multiple images
- GeoJSON output for map integration
- Overlay visualization generation
- Health check endpoints
- CORS enabled for frontend integration

## Setup

### 1. Install Dependencies

\`\`\`bash
cd backend
pip install -r ../requirements.txt
\`\`\`

### 2. Ensure Model is Trained

The API requires a trained model at `models/flood_segmentation_model.h5` (from project root)

If you haven't trained the model yet:
\`\`\`bash
cd ..
python scripts/train_unet_model.py
\`\`\`

### 3. Start the Server

**Option A: From Root Directory (Recommended)**
\`\`\`bash
python backend/app.py
\`\`\`

**Option B: From Backend Directory**
\`\`\`bash
cd backend
MODEL_PATH=../models/flood_segmentation_model.h5 python app.py
\`\`\`

Server will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check

**GET** `/api/health`

Check if the server is running and model is loaded.

**Response:**
\`\`\`json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/flood_segmentation_model.h5",
  "timestamp": "2025-01-20T10:30:00"
}
\`\`\`

### 2. Predict Flood Segmentation

**POST** `/api/predict`

Upload an image and get flood segmentation prediction.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` (file)

**Response:**
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
  },
  "metadata": {
    "original_size": [1024, 768],
    "processed_size": [256, 256],
    "threshold": 0.5,
    "timestamp": "2025-01-20T10:30:00"
  }
}
\`\`\`

### 3. Predict with Coordinates (for Maps)

**POST** `/api/predict-coordinates`

Get flood predictions in GeoJSON format for map visualization.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - `image` (file)
  - `bounds` (string, optional): "lat1,lng1,lat2,lng2"

**Response:**
\`\`\`json
{
  "success": true,
  "prediction": {
    "flood_detected": true,
    "flood_percentage": 23.5,
    "risk_level": "High"
  },
  "geojson": {
    "type": "FeatureCollection",
    "features": [...]
  },
  "metadata": {
    "num_flood_zones": 3,
    "timestamp": "2025-01-20T10:30:00"
  }
}
\`\`\`

### 4. Batch Prediction

**POST** `/api/batch-predict`

Process multiple images in one request.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `images[]` (multiple files)

**Response:**
\`\`\`json
{
  "success": true,
  "total_images": 3,
  "results": [
    {
      "filename": "image1.jpg",
      "success": true,
      "flood_detected": true,
      "flood_percentage": 15.3,
      "risk_level": "Medium"
    },
    ...
  ]
}
\`\`\`

## Testing

Use the provided test script:

\`\`\`bash
# Start the server first in another terminal
python backend/app.py

# In another terminal, run tests
python backend/test_api.py
\`\`\`

Or use curl:

\`\`\`bash
# Health check
curl http://localhost:5000/api/health

# Prediction
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/api/predict

# Coordinate prediction
curl -X POST -F "image=@test_image.jpg" -F "bounds=40.7128,-74.0060,40.7828,-73.9260" http://localhost:5000/api/predict-coordinates
\`\`\`

## Risk Levels

The API categorizes flood risk into 4 levels based on flood percentage:

- **Low**: < 5% flooded area
- **Medium**: 5-15% flooded area
- **High**: 15-30% flooded area
- **Critical**: > 30% flooded area

## Configuration

Environment variables (optional):

- `MODEL_PATH`: Path to trained model (default: `models/flood_segmentation_model.h5` from root)

## Troubleshooting

### Model Not Found
- Ensure you've trained the model first using `python scripts/train_unet_model.py`
- Check that the model file exists at `models/flood_segmentation_model.h5`
- If running from backend directory, use: `MODEL_PATH=../models/flood_segmentation_model.h5 python app.py`

### Out of Memory
- Reduce image size before uploading
- Use batch prediction with smaller batches

### CORS Issues
- CORS is enabled for all origins by default
- Modify `CORS(app)` in app.py if you need specific origins

## Next Steps

After setting up the backend, proceed to build the frontend with Leaflet integration to visualize predictions on interactive maps.
