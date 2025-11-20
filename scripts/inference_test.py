"""
Standalone inference test script
Tests the trained model on sample images without needing the web server
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
MODEL_PATH = "../models/flood_segmentation_model.h5"
IMG_HEIGHT = 256
IMG_WIDTH = 256
CONFIDENCE_THRESHOLD = 0.5


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


def load_model(model_path):
    """Load the trained model"""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python scripts/train_unet_model.py")
        return None
    
    try:
        custom_objects = {
            'dice_coefficient': dice_coefficient,
            'dice_loss': dice_loss,
            'combined_loss': combined_loss,
            'iou_metric': iou_metric
        }
        
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image_path):
    """Preprocess image for model input"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Resize
    image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch, image, image_resized


def predict(model, image_path):
    """Run inference on a single image"""
    print(f"\nProcessing: {image_path}")
    
    # Preprocess
    image_batch, original_image, resized_image = preprocess_image(image_path)
    
    # Predict
    prediction = model.predict(image_batch, verbose=0)
    
    # Extract mask
    pred_mask = prediction[0, :, :, 0]
    
    # Calculate statistics
    flood_percentage = np.mean(pred_mask > CONFIDENCE_THRESHOLD) * 100
    max_confidence = np.max(pred_mask)
    avg_confidence = np.mean(pred_mask[pred_mask > CONFIDENCE_THRESHOLD]) if np.any(pred_mask > CONFIDENCE_THRESHOLD) else 0
    
    # Binary mask
    binary_mask = (pred_mask > CONFIDENCE_THRESHOLD).astype(np.uint8) * 255
    
    # Risk level
    if flood_percentage < 5:
        risk_level = 'Low'
    elif flood_percentage < 15:
        risk_level = 'Medium'
    elif flood_percentage < 30:
        risk_level = 'High'
    else:
        risk_level = 'Critical'
    
    results = {
        'flood_percentage': flood_percentage,
        'max_confidence': max_confidence,
        'avg_confidence': avg_confidence,
        'risk_level': risk_level,
        'pred_mask': pred_mask,
        'binary_mask': binary_mask,
        'original_image': original_image,
        'resized_image': resized_image
    }
    
    return results


def visualize_results(results, save_path=None):
    """Visualize prediction results"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(results['resized_image'])
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Probability mask
    axes[1].imshow(results['pred_mask'], cmap='hot')
    axes[1].set_title('Probability Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Binary mask
    axes[2].imshow(results['binary_mask'], cmap='Blues')
    axes[2].set_title('Binary Mask (Threshold=0.5)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Overlay
    overlay = results['resized_image'].copy()
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = results['pred_mask'] * 255
    overlay = cv2.addWeighted(overlay, 0.6, red_mask.astype(np.uint8), 0.4, 0)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    # Add statistics
    stats_text = f"Flood Coverage: {results['flood_percentage']:.2f}%\n"
    stats_text += f"Risk Level: {results['risk_level']}\n"
    stats_text += f"Max Confidence: {results['max_confidence']:.3f}\n"
    stats_text += f"Avg Confidence: {results['avg_confidence']:.3f}"
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def test_single_image(model, image_path):
    """Test on a single image"""
    results = predict(model, image_path)
    
    print("\nResults:")
    print(f"  Flood Coverage: {results['flood_percentage']:.2f}%")
    print(f"  Risk Level: {results['risk_level']}")
    print(f"  Max Confidence: {results['max_confidence']:.3f}")
    print(f"  Avg Confidence: {results['avg_confidence']:.3f}")
    
    # Save visualization
    output_path = f"inference_result_{Path(image_path).stem}.png"
    visualize_results(results, output_path)


def test_batch_images(model, image_folder):
    """Test on multiple images in a folder"""
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(Path(image_folder).glob(ext))
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return
    
    print(f"\nFound {len(image_paths)} images")
    
    results_summary = []
    
    for image_path in image_paths:
        try:
            results = predict(model, str(image_path))
            results_summary.append({
                'filename': image_path.name,
                'flood_percentage': results['flood_percentage'],
                'risk_level': results['risk_level']
            })
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PREDICTION SUMMARY")
    print("="*60)
    print(f"{'Filename':<30} {'Flood %':<12} {'Risk Level':<15}")
    print("-"*60)
    
    for result in results_summary:
        print(f"{result['filename']:<30} {result['flood_percentage']:>8.2f}%   {result['risk_level']:<15}")
    
    print("="*60)


def main():
    """Main function"""
    print("="*60)
    print("Flood Segmentation Inference Test")
    print("="*60)
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  Single image: python inference_test.py <image_path>")
        print("  Batch images: python inference_test.py <folder_path>")
        print("\nExample:")
        print("  python inference_test.py test_image.jpg")
        print("  python inference_test.py test_images/")
        return
    
    path = sys.argv[1]
    
    if os.path.isfile(path):
        # Single image
        test_single_image(model, path)
    elif os.path.isdir(path):
        # Batch processing
        test_batch_images(model, path)
    else:
        print(f"Error: Path not found: {path}")


if __name__ == "__main__":
    main()
