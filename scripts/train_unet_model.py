"""
U-Net Model Training Script for Flood Segmentation
This script trains a U-Net model to segment flood zones from satellite/aerial imagery
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

# Paths (update these based on your dataset location)
DATASET_PATH = "data/flood_dataset"  # Update this path
IMAGES_PATH = os.path.join(DATASET_PATH, "images")
MASKS_PATH = os.path.join(DATASET_PATH, "masks")
MODEL_SAVE_PATH = "models/flood_segmentation_model.h5"


def build_unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    """
    Build U-Net architecture for semantic segmentation
    U-Net consists of:
    - Encoder (downsampling path): captures context
    - Decoder (upsampling path): enables precise localization
    - Skip connections: concatenate encoder features to decoder
    """
    
    inputs = keras.Input(shape=input_shape)
    
    # Encoder (Contracting Path)
    # Block 1
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # Block 2
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Block 3
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # Block 4
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder (Expanding Path)
    # Block 6
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])  # Skip connection
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    # Block 7
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])  # Skip connection
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    # Block 8
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])  # Skip connection
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    # Block 9
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])  # Skip connection
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    # Output layer - sigmoid activation for binary segmentation
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs], name='U-Net')
    
    return model


def load_image(image_path):
    """Load and preprocess image"""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=IMG_CHANNELS)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
    return img


def load_mask(mask_path):
    """Load and preprocess mask"""
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])
    mask = tf.cast(mask, tf.float32) / 255.0  # Normalize to [0, 1]
    return mask


def dice_coefficient(y_true, y_pred, smooth=1):
    """
    Dice coefficient metric for segmentation
    Measures overlap between predicted and true masks
    """
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
    """
    Intersection over Union (IoU) metric
    Standard metric for segmentation tasks
    """
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)


def create_dataset(image_paths, mask_paths, batch_size=BATCH_SIZE, augment=False):
    """Create TensorFlow dataset from image and mask paths"""
    
    def load_data(image_path, mask_path):
        image = load_image(image_path)
        mask = load_mask(mask_path)
        return image, mask
    
    def augment_data(image, mask):
        """Data augmentation for training"""
        # Random flip
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        # Random rotation (90, 180, 270 degrees)
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        mask = tf.image.rot90(mask, k)
        
        return image, mask
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def visualize_predictions(model, test_dataset, num_samples=3):
    """Visualize model predictions"""
    plt.figure(figsize=(15, 5 * num_samples))
    
    for idx, (images, masks) in enumerate(test_dataset.take(1)):
        predictions = model.predict(images)
        
        for i in range(min(num_samples, len(images))):
            # Original image
            plt.subplot(num_samples, 3, idx * num_samples + i * 3 + 1)
            plt.imshow(images[i])
            plt.title('Original Image')
            plt.axis('off')
            
            # Ground truth mask
            plt.subplot(num_samples, 3, idx * num_samples + i * 3 + 2)
            plt.imshow(masks[i].numpy().squeeze(), cmap='Blues')
            plt.title('Ground Truth')
            plt.axis('off')
            
            # Predicted mask
            plt.subplot(num_samples, 3, idx * num_samples + i * 3 + 3)
            plt.imshow(predictions[i].squeeze(), cmap='Blues')
            plt.title('Prediction')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    print("Visualization saved as 'predictions_visualization.png'")


def main():
    """Main training function"""
    
    print("=" * 50)
    print("Flood Segmentation U-Net Training")
    print("=" * 50)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/flood_dataset/images", exist_ok=True)
    os.makedirs("data/flood_dataset/masks", exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(IMAGES_PATH) or len(os.listdir(IMAGES_PATH)) == 0:
        print("\n⚠️  WARNING: No dataset found!")
        print(f"Please add your dataset to: {DATASET_PATH}")
        print("\nDataset structure should be:")
        print("data/flood_dataset/")
        print("  ├── images/  (satellite/aerial images)")
        print("  └── masks/   (binary flood masks)")
        print("\nRecommended datasets:")
        print("1. Kaggle: 'Flood Area Segmentation'")
        print("2. Sen1Floods11 dataset")
        print("3. Roboflow Universe flood datasets")
        return
    
    # Load dataset paths
    image_files = sorted([os.path.join(IMAGES_PATH, f) for f in os.listdir(IMAGES_PATH) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([os.path.join(MASKS_PATH, f) for f in os.listdir(MASKS_PATH) 
                        if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\n📊 Dataset Statistics:")
    print(f"Total images: {len(image_files)}")
    print(f"Total masks: {len(mask_files)}")
    
    if len(image_files) != len(mask_files):
        print("\n❌ Error: Number of images and masks don't match!")
        return
    
    # Split dataset
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_files, mask_files, test_size=0.2, random_state=42
    )
    
    print(f"\n📚 Split:")
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Create datasets
    print("\n🔨 Creating datasets...")
    train_dataset = create_dataset(train_images, train_masks, augment=True)
    val_dataset = create_dataset(val_images, val_masks, augment=False)
    
    # Build model
    print("\n🏗️  Building U-Net model...")
    model = build_unet_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=[dice_coefficient, iou_metric, 'accuracy']
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    print("\n📈 Plotting training history...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['dice_coefficient'], label='Training Dice')
    plt.plot(history.history['val_dice_coefficient'], label='Validation Dice')
    plt.title('Dice Coefficient over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['iou_metric'], label='Training IoU')
    plt.plot(history.history['val_iou_metric'], label='Validation IoU')
    plt.title('IoU Metric over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved as 'training_history.png'")
    
    # Visualize predictions
    print("\n🎨 Generating prediction visualizations...")
    visualize_predictions(model, val_dataset)
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"\n📊 Final Metrics:")
    print(f"Validation Dice Coefficient: {history.history['val_dice_coefficient'][-1]:.4f}")
    print(f"Validation IoU: {history.history['val_iou_metric'][-1]:.4f}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
