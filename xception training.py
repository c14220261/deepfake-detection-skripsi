import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

# path data
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'dataset', 'frames_C23')
model_path = os.path.join(base_dir, 'deepfake_detector_xception.h5')

print("Versi TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Terdeteksi: {len(gpus)}")
    except RuntimeError as e:
        print(f"Error konfigurasi GPU: {e}")
else:
    print("Num GPUs Available: 0 (Menggunakan CPU)")

# Preprocessing
IMG_SIZE = (299, 299)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Membangun Model Xception
def build_xception_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  # Freeze layer awal

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

if os.path.exists(model_path):
    print(f"\n[INFO] Menemukan model tersimpan di {model_path}. Memuat model untuk melanjutkan...")
    model = tf.keras.models.load_model(model_path)
else:
    print("\n[INFO] Tidak ada model tersimpan. Membangun model baru dari awal...")
    model = build_xception_model()

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,   # Hanya simpan jika hasil validasi membaik
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# Proses Training
EPOCHS = 10
print("\nMemulai training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback] # Auto-save setiap epoch
)

# Fine-Tuning
print("\nMelakukan Fine-tuning untuk meningkatkan akurasi...")
model.layers[0].trainable = True # Unfreeze semua layer
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(train_generator, epochs=5, validation_data=validation_generator, callbacks=[checkpoint_callback])

# Visualisasi
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.legend()
plt.show()