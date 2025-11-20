"""
Flask Backend API for Flood Segmentation System
Handles image uploads, model inference, and prediction serving
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'models/flood_segmentation_model.h5')
IMG_HEIGHT = 256
IMG_WIDTH = 256
CONFIDENCE_THRESHOLD = 0.5

# Global model variable
model = None


def load_model():
    """Load the trained U-Net model"""
    global model
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        return False
    
    try:
        # Custom objects for model loading
        custom_objects = {
            'dice_coefficient': dice_coefficient,
            'dice_loss': dice_loss,
            'combined_loss': combined_loss,
            'iou_metric': iou_metric
        }
        
        model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


# Metrics (must match training script)
def dice_coefficient(y_true, y_pred, smooth=1):
    """Dice coefficient metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """Combination of binary crossentropy and dice loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred, threshold=0.5):
    """Intersection over Union metric"""
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)


def preprocess_image(image_data):
    """
    Preprocess uploaded image for model input
    Args:
        image_data: PIL Image or numpy array
    Returns:
        preprocessed numpy array ready for model
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image_data, Image.Image):
        image_data = np.array(image_data)
    
    # Resize to model input size
    image_resized = cv2.resize(image_data, (IMG_WIDTH, IMG_HEIGHT))
    
    # Ensure 3 channels
    if len(image_resized.shape) == 2:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    elif image_resized.shape[2] == 4:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGBA2RGB)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch, image_resized


def postprocess_prediction(prediction, original_size=None):
    """
    Postprocess model prediction
    Args:
        prediction: model output (batch, height, width, 1)
        original_size: tuple (width, height) to resize prediction
    Returns:
        binary mask and confidence scores
    """
    # Remove batch dimension and channel dimension
    pred_mask = prediction[0, :, :, 0]
    
    # Resize to original size if specified
    if original_size is not None:
        pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_LINEAR)
    
    # Calculate statistics
    flood_percentage = np.mean(pred_mask > CONFIDENCE_THRESHOLD) * 100
    max_confidence = np.max(pred_mask)
    avg_confidence = np.mean(pred_mask[pred_mask > CONFIDENCE_THRESHOLD]) if np.any(pred_mask > CONFIDENCE_THRESHOLD) else 0
    
    # Create binary mask
    binary_mask = (pred_mask > CONFIDENCE_THRESHOLD).astype(np.uint8) * 255
    
    return {
        'mask': pred_mask,
        'binary_mask': binary_mask,
        'flood_percentage': float(flood_percentage),
        'max_confidence': float(max_confidence),
        'avg_confidence': float(avg_confidence),
        'flood_detected': flood_percentage > 1.0
    }


def mask_to_base64(mask):
    """Convert numpy mask to base64 encoded PNG"""
    # Convert to PIL Image
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
    img = Image.fromarray(mask_uint8)
    
    # Save to buffer
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_base64


def create_overlay(original_image, mask, alpha=0.5):
    """
    Create overlay visualization of flood mask on original image
    Args:
        original_image: numpy array (H, W, 3)
        mask: numpy array (H, W) with values [0, 1]
        alpha: transparency of overlay
    Returns:
        overlayed image as numpy array
    """
    # Ensure mask is same size as image
    if mask.shape[:2] != original_image.shape[:2]:
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    
    # Create colored mask (red for flood areas)
    colored_mask = np.zeros_like(original_image)
    colored_mask[:, :, 0] = mask * 255  # Red channel
    
    # Blend original image with mask
    overlay = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)
    
    return overlay


@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'Flood Segmentation API',
        'version': '1.0.0',
        'model_loaded': model is not None
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts an image file and returns flood segmentation prediction
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train model first.'}), 500
    
    # Check if image file is in request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        original_size = image.size  # (width, height)
        
        logger.info(f"Processing image: {file.filename}, size: {original_size}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess for model
        image_batch, image_resized = preprocess_image(image)
        
        # Run inference
        logger.info("Running model inference...")
        prediction = model.predict(image_batch, verbose=0)
        
        # Postprocess prediction
        result = postprocess_prediction(prediction, original_size)
        
        # Convert masks to base64 for JSON response
        mask_base64 = mask_to_base64(result['mask'])
        binary_mask_base64 = mask_to_base64(result['binary_mask'])
        
        # Create overlay visualization
        original_array = np.array(image)
        overlay = create_overlay(original_array, result['mask'])
        overlay_base64 = mask_to_base64(overlay)
        
        logger.info(f"Prediction complete: {result['flood_percentage']:.2f}% flood detected")
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'flood_detected': result['flood_detected'],
                'flood_percentage': result['flood_percentage'],
                'max_confidence': result['max_confidence'],
                'avg_confidence': result['avg_confidence'],
                'risk_level': get_risk_level(result['flood_percentage'])
            },
            'masks': {
                'probability_mask': f"data:image/png;base64,{mask_base64}",
                'binary_mask': f"data:image/png;base64,{binary_mask_base64}",
                'overlay': f"data:image/png;base64,{overlay_base64}"
            },
            'metadata': {
                'original_size': original_size,
                'processed_size': (IMG_WIDTH, IMG_HEIGHT),
                'threshold': CONFIDENCE_THRESHOLD,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/predict-coordinates', methods=['POST'])
def predict_coordinates():
    """
    Prediction endpoint that returns flood coordinates for map visualization
    Returns GeoJSON format for Leaflet integration
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Get optional map bounds from request
    bounds = request.form.get('bounds')  # Expected format: "lat1,lng1,lat2,lng2"
    
    try:
        file = request.files['image']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess and predict
        image_batch, _ = preprocess_image(image)
        prediction = model.predict(image_batch, verbose=0)
        result = postprocess_prediction(prediction, image.size)
        
        # Convert binary mask to contours for GeoJSON
        contours, _ = cv2.findContours(
            result['binary_mask'],
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert contours to GeoJSON format
        features = []
        
        if bounds:
            # Parse bounds: "lat1,lng1,lat2,lng2"
            coords = [float(x) for x in bounds.split(',')]
            lat1, lng1, lat2, lng2 = coords
            
            # Map pixel coordinates to geographic coordinates
            height, width = result['binary_mask'].shape
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small contours
                    # Convert pixel coordinates to lat/lng
                    polygon_coords = []
                    for point in contour:
                        x, y = point[0]
                        lng = lng1 + (x / width) * (lng2 - lng1)
                        lat = lat1 + (y / height) * (lat2 - lat1)
                        polygon_coords.append([lng, lat])
                    
                    # Close the polygon
                    if len(polygon_coords) > 0:
                        polygon_coords.append(polygon_coords[0])
                        
                        features.append({
                            'type': 'Feature',
                            'geometry': {
                                'type': 'Polygon',
                                'coordinates': [polygon_coords]
                            },
                            'properties': {
                                'flood': True,
                                'confidence': float(result['avg_confidence'])
                            }
                        })
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        response = {
            'success': True,
            'prediction': {
                'flood_detected': result['flood_detected'],
                'flood_percentage': result['flood_percentage'],
                'risk_level': get_risk_level(result['flood_percentage'])
            },
            'geojson': geojson,
            'metadata': {
                'num_flood_zones': len(features),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during coordinate prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


def get_risk_level(flood_percentage):
    """Determine risk level based on flood percentage"""
    if flood_percentage < 5:
        return 'Low'
    elif flood_percentage < 15:
        return 'Medium'
    elif flood_percentage < 30:
        return 'High'
    else:
        return 'Critical'


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple images
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'images' not in request.files:
        return jsonify({'error': 'No image files provided'}), 400
    
    files = request.files.getlist('images')
    
    if len(files) == 0:
        return jsonify({'error': 'Empty file list'}), 400
    
    results = []
    
    for file in files:
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_batch, _ = preprocess_image(image)
            prediction = model.predict(image_batch, verbose=0)
            result = postprocess_prediction(prediction)
            
            results.append({
                'filename': file.filename,
                'success': True,
                'flood_detected': result['flood_detected'],
                'flood_percentage': result['flood_percentage'],
                'risk_level': get_risk_level(result['flood_percentage'])
            })
        
        except Exception as e:
            results.append({
                'filename': file.filename,
                'success': False,
                'error': str(e)
            })
    
    return jsonify({
        'success': True,
        'total_images': len(files),
        'results': results
    })


if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting Flood Segmentation API...")
    
    if load_model():
        logger.info("Model loaded successfully. Starting server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load model. Please train model first using scripts/train_unet_model.py")
        logger.info("Starting server anyway for testing purposes...")
        app.run(debug=True, host='0.0.0.0', port=5000)
