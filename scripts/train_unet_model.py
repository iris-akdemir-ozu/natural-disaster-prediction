"""
U-Net Model Training Script for Multi-Class Disaster Segmentation
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ================= AYARLAR =================
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
NUM_CLASSES = 5  # xView2 verisi için sınıf sayısı (0:Yok, 1-4: Hasar Seviyeleri)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

DATASET_PATH = "data/flood_dataset"
IMAGES_PATH = os.path.join(DATASET_PATH, "images")
MASKS_PATH = os.path.join(DATASET_PATH, "masks")
MODEL_SAVE_PATH = "models/flood_segmentation_model.h5"
# ===========================================

def build_unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    # Output Layer (Multi-class Softmax)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs], name='U-Net_MultiClass')
    return model

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def load_mask(mask_path):
    """
    Maske Yükleme (Multi-Class için Güncellendi)
    - 255'e BÖLMÜYORUZ (Sınıf ID'leri kaybolmasın diye)
    - One-Hot Encoding yapıyoruz.
    """
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    
    # Piksel değerlerini bozmadan yeniden boyutlandır (Nearest Neighbor önemli!)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
    
    # Sınıf numaraları integer olmalı
    mask = tf.cast(mask, tf.int32)
    
    # One-Hot Encoding (Örn: 2 -> [0,0,1,0,0])
    # Çıktı şekli: (256, 256, 1, 5) -> Squeeze ile (256, 256, 5) yapıyoruz
    mask = tf.one_hot(mask, NUM_CLASSES)
    mask = tf.squeeze(mask, axis=2) 
    
    return mask

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    # Binary Crossentropy -> Categorical Crossentropy olarak değişti
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return cce + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32) # Threshold
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def create_dataset(image_paths, mask_paths, batch_size=BATCH_SIZE, augment=False):
    def load_data(image_path, mask_path):
        image = load_image(image_path)
        mask = load_mask(mask_path)
        return image, mask
    
    # Basitleştirilmiş dataset pipeline (Hata riskini azaltmak için paralelliği kapattık)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_data, num_parallel_calls=1) 
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    
    return dataset

def visualize_predictions(model, test_dataset, num_samples=3):
    plt.figure(figsize=(15, 5 * num_samples))
    for idx, (images, masks) in enumerate(test_dataset.take(1)):
        predictions = model.predict(images)
        for i in range(min(num_samples, len(images))):
            # Orijinal
            plt.subplot(num_samples, 3, i * 3 + 1)
            plt.imshow(images[i])
            plt.title('Original')
            plt.axis('off')
            
            # Gerçek Maske (One-hot'tan geri çeviriyoruz göstermek için)
            true_mask = tf.argmax(masks[i], axis=-1)
            plt.subplot(num_samples, 3, i * 3 + 2)
            plt.imshow(true_mask, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
            plt.title('True Mask')
            plt.axis('off')
            
            # Tahmin
            pred_mask = tf.argmax(predictions[i], axis=-1)
            plt.subplot(num_samples, 3, i * 3 + 3)
            plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
            plt.title('Prediction')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    print("Visualization saved.")

def main():
    print("=" * 50)
    print(f"Multi-Class ({NUM_CLASSES}) U-Net Training")
    print("=" * 50)
    
    # Klasör Kontrolü
    if not os.path.exists(IMAGES_PATH) or not os.listdir(IMAGES_PATH):
        print("❌ Dataset not found!")
        return

    # Dosyaları Yükle
    image_files = sorted([os.path.join(IMAGES_PATH, f) for f in os.listdir(IMAGES_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([os.path.join(MASKS_PATH, f) for f in os.listdir(MASKS_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Eşleşme Kontrolü
    # Not: xView2 datasetinde maske isimleri bazen farklı olabilir, burada basit sıralama varsayıyoruz.
    # Eğer prepare_dataset.py kullandıysanız isimler zaten aynıdır.
    
    print(f"Images: {len(image_files)} | Masks: {len(mask_files)}")
    
    # Eğitim/Test Ayrımı
    train_images, val_images, train_masks, val_masks = train_test_split(image_files, mask_files, test_size=0.2, random_state=42)
    
    train_dataset = create_dataset(train_images, train_masks)
    val_dataset = create_dataset(val_images, val_masks)
    
    # Model Kurulumu
    model = build_unet_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=[dice_coefficient, iou_metric, 'accuracy']
    )
    
    # Eğitim
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss'),
            keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')
        ]
    )
    
    visualize_predictions(model, val_dataset)
    print("✅ Training Finished!")

if __name__ == "__main__":
    main()