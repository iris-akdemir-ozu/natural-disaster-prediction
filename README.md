# Flood Segmentation Deep Learning System

AI-powered natural disaster prediction system using U-Net deep learning for flood zone identification from satellite and aerial imagery.

## Overview

This system uses a U-Net deep learning model to perform semantic segmentation on satellite or aerial images to identify flood-affected areas. The project consists of three main components:

1. **Deep Learning Model** - U-Net architecture for flood segmentation
2. **Backend API** - Flask server for model inference
3. **Frontend** - Interactive web interface with Leaflet.js map integration

## Project Structure

\`\`\`
flood-segmentation-system/
├── scripts/
│   ├── train_unet_model.py          # Model training script
│   ├── inference_test.py            # Standalone inference testing
│   └── sample_dataset.py            # Dataset setup helper
├── backend/
│   ├── app.py                       # Flask API server
│   └── test_api.py                  # API testing script
├── models/
│   └── flood_segmentation_model.h5  # Trained model (after training)
├── data/
│   └── flood_dataset/
│       ├── images/                  # Training images
│       └── masks/                   # Ground truth masks
├── src/
│   ├── routes/
│   │   └── +page.svelte            # Main page
│   └── lib/
│       ├── components/              # UI components
│       └── stores/                  # State management
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
└── package.json                     # Node.js dependencies
\`\`\`

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- 2GB+ RAM (4GB+ recommended)
- GPU recommended for training (optional)

### 1. Setup Python Environment

\`\`\`bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
\`\`\`

### 2. Prepare Dataset

Download a flood segmentation dataset:

**Option A: Kaggle (Quick Start)**
- Search "Flood Area Segmentation" on Kaggle
- Download and extract to `data/flood_dataset/`

**Option B: Sen1Floods11 (Professional)**
- High-quality dataset with 4,831 images
- Search "Sen1Floods11" on Google or Hugging Face

**Dataset Structure:**
\`\`\`
data/flood_dataset/
├── images/     # Satellite/aerial images
│   ├── img_001.png
│   └── ...
└── masks/      # Binary flood masks (white=flood, black=no flood)
    ├── mask_001.png
    └── ...
\`\`\`

### 3. Train the Model

\`\`\`bash
python scripts/train_unet_model.py
\`\`\`

Training will:
- Split data 80/20 (train/validation)
- Apply data augmentation
- Train for 50 epochs with early stopping
- Save best model to `models/flood_segmentation_model.h5`
- Generate training history plots

Expected training time:
- Small dataset (100-300 images): 10-30 min (CPU) / 2-5 min (GPU)
- Large dataset (2000+ images): 2-5 hours (CPU) / 30-60 min (GPU)

### 4. Start Backend Server

\`\`\`bash
# Run from project root (recommended)
python backend/app.py
\`\`\`

Backend will be available at `http://localhost:5000`

**Note:** The backend will start even if no model is found, but `/api/predict` won't work until you train the model.

### 5. Start Frontend

\`\`\`bash
# Install Node.js dependencies
npm install

# Start development server
npm run dev
\`\`\`

Frontend will be available at `http://localhost:5173`

### 6. Use the Application

1. Open browser to `http://localhost:5173`
2. Click "Upload Image" and select a satellite/aerial image
3. Click "Analyze Image"
4. View results: risk level, flood coverage, and interactive map

## API Usage

### Health Check

\`\`\`bash
curl http://localhost:5000/api/health
\`\`\`

### Single Prediction

\`\`\`bash
curl -X POST \
  http://localhost:5000/api/predict \
  -F "image=@test_image.jpg"
\`\`\`

### Batch Prediction

\`\`\`bash
curl -X POST \
  http://localhost:5000/api/batch-predict \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg"
\`\`\`

### Python Client

\`\`\`python
import requests

with open('test_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)
    
result = response.json()
print(f"Risk level: {result['prediction']['risk_level']}")
print(f"Flood coverage: {result['prediction']['flood_percentage']:.2f}%")
\`\`\`

## Model Details

### U-Net Architecture

- **Encoder**: 4 convolutional blocks (64, 128, 256, 512 filters)
- **Bottleneck**: 1024 filters
- **Decoder**: 4 transposed convolutional blocks with skip connections
- **Output**: Single channel with sigmoid activation

### Training Configuration

- Image size: 256x256
- Batch size: 16
- Optimizer: Adam (lr=0.0001)
- Loss: Combined Binary Cross-Entropy + Dice Loss
- Metrics: Dice Coefficient, IoU, Accuracy

### Performance Metrics

Good model indicators:
- Dice Coefficient > 0.75
- IoU > 0.65
- Validation loss < 0.3

## Standalone Inference Testing

Test the model without the web interface:

\`\`\`bash
# Single image
python scripts/inference_test.py test_image.jpg

# Batch processing
python scripts/inference_test.py test_images_folder/
\`\`\`

## Building for Production

To create a production build of the frontend:

\`\`\`bash
npm run build
\`\`\`

Preview the production build:

\`\`\`bash
npm run preview
\`\`\`

For deployment instructions, see `DEPLOYMENT.md`.

## Troubleshooting

### "Model Not Found" Error

If you see:
\`\`\`
ERROR:__main__:Model not found at models/flood_segmentation_model.h5
ERROR:__main__:Failed to load model. Please train model first
\`\`\`

**This is normal before training!** You need to:
1. Download a dataset: `python scripts/sample_dataset.py` (then manually download from Kaggle/Roboflow)
2. Train the model: `python scripts/train_unet_model.py`
3. Restart the backend: `python backend/app.py`

The backend starts anyway for testing, but predictions require the trained model.

### "No Dataset Found" Warning

If you see:
\`\`\`
WARNING: No dataset found!
Please add your dataset to: data/flood_dataset
\`\`\`

You need to manually download a flood segmentation dataset and place it in `data/flood_dataset/` with the structure:
\`\`\`
data/flood_dataset/
├── images/     # Your image files here
└── masks/      # Corresponding mask files here
\`\`\`

### Model Not Found

- Ensure you've trained the model first: `python scripts/train_unet_model.py`
- Check that `models/flood_segmentation_model.h5` exists

### Out of Memory During Training

- Reduce `BATCH_SIZE` in training script (try 8 or 4)
- Reduce image dimensions (try 128x128)

### Backend Connection Failed

- Ensure Flask server is running: `python backend/app.py`
- Check if port 5000 is not blocked by firewall
- Verify API URL in frontend matches backend

### Poor Predictions

- Increase training dataset size (aim for 500+ images)
- Verify masks are correctly labeled (white=flood, black=no flood)
- Train for more epochs
- Use higher quality source imagery

## Documentation

- `README.md` - This file (project overview and quick start)
- `USAGE_GUIDE.md` - Complete usage guide with examples
- `DEPLOYMENT.md` - Production deployment instructions
- `backend/README.md` - Backend API documentation

## Features

- Deep learning-based flood segmentation using U-Net
- RESTful API with multiple prediction endpoints
- Interactive web interface with Leaflet.js maps
- Real-time flood zone visualization
- Risk level assessment (Low/Medium/High/Critical)
- Batch processing support
- Standalone inference tools
- Comprehensive documentation

## Technology Stack

**Backend:**
- Python 3.8+
- TensorFlow/Keras 2.13+
- Flask 3.0+
- OpenCV, NumPy, Pillow

**Frontend:**
- SvelteKit 5+
- Leaflet.js 1.9+
- TailwindCSS 4+
- TypeScript

**Model:**
- U-Net architecture
- Semantic segmentation
- Binary classification (flood/no flood)

## License

This project is for educational and research purposes.

## Support

For detailed usage instructions, see `USAGE_GUIDE.md`.
For deployment help, see `DEPLOYMENT.md`.

## Acknowledgments

- U-Net architecture from Ronneberger et al. (2015)
- Dataset sources: Kaggle, Sen1Floods11, Roboflow Universe
- Built with TensorFlow, Flask, SvelteKit, and Leaflet.js
