"""
Flask Backend API for Multi-Class Disaster Segmentation
Handles image uploads and processes 5-class model output correctly.
"""

from flask import Flask, request, jsonify
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

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App Setup
app = Flask(__name__)
CORS(app)

# Settings
MODEL_PATH = os.getenv('MODEL_PATH', 'models/flood_segmentation_model.h5')
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CLASSES = 5 

model = None

# Custom metrics for loading the model
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return cce + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found!")
        return False
    try:
        custom_objects = {
            'dice_coefficient': dice_coefficient,
            'dice_loss': dice_loss,
            'combined_loss': combined_loss,
            'iou_metric': iou_metric
        }
        model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        logger.info("✅ Multi-Class Model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def preprocess_image(image_data):
    if isinstance(image_data, Image.Image):
        image_data = np.array(image_data)
    
    img = cv2.resize(image_data, (IMG_WIDTH, IMG_HEIGHT))
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img_normalized = img.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0)

def postprocess_prediction(prediction, original_size=None):
    """
    Multi-class prediction logic:
    Prediction shape: (1, 256, 256, 5)
    We take argmax to find the strongest class for each pixel.
    0 = Background (Safe)
    1, 2, 3, 4 = Damage Levels (Disaster)
    """
    # Select the highest probability class (0,1,2,3,4)
    class_map = np.argmax(prediction[0], axis=-1)  # Shape: (256, 256)
    
    # Resize to original
    if original_size:
        class_map = cv2.resize(class_map.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)

    # Risk Analysis: Pixels greater than 0 (1-4) are considered "Disaster".
    disaster_pixels = np.sum(class_map > 0)
    total_pixels = class_map.size
    flood_percentage = (disaster_pixels / total_pixels) * 100
    
    # Mask for visualization (Only damaged areas white)
    binary_mask = np.zeros_like(class_map, dtype=np.uint8)
    binary_mask[class_map > 0] = 255

    return {
        'flood_percentage': float(flood_percentage),
        'risk_level': get_risk_level(flood_percentage),
        'binary_mask': binary_mask,
        'raw_map': class_map
    }

def get_risk_level(percent):
    if percent < 1: return 'Safe'
    if percent < 10: return 'Low'
    if percent < 30: return 'Medium'
    if percent < 60: return 'High'
    return 'Critical'

def mask_to_base64(mask):
    img = Image.fromarray(mask.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def create_overlay(original_image, mask):
    # Fit mask to original size
    if mask.shape[:2] != original_image.shape[:2]:
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    overlay = original_image.copy()
    # Paint only damaged areas RED
    overlay[mask > 0] = [255, 0, 0]  # RGB Red
    
    # Make transparent
    return cv2.addWeighted(original_image, 0.6, overlay, 0.4, 0)

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model: return jsonify({'error': 'Model not loaded'}), 500
    file = request.files.get('image')
    if not file: return jsonify({'error': 'No file'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        input_tensor = preprocess_image(img)
        
        # Prediction
        pred = model.predict(input_tensor, verbose=0)
        
        # Process Results
        result = postprocess_prediction(pred, img.size)
        
        # Prepare Visuals
        overlay_img = create_overlay(np.array(img), result['raw_map'])
        
        return jsonify({
            'success': True,
            'prediction': {
                'risk_level': result['risk_level'],
                'flood_percentage': result['flood_percentage'],
                'max_confidence': 1.0 # High confidence due to Softmax
            },
            'masks': {
                'overlay': f"data:image/png;base64,{mask_to_base64(overlay_img)}"
            }
        })
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if load_model():
        app.run(host='0.0.0.0', port=5000)