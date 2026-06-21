import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = "deepfake_detector_xception.h5"
DATASET_PATH = r'dataset\frames_C40'
IMG_SIZE = (299, 299)
BATCH_SIZE = 32

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model {MODEL_PATH} tidak ditemukan!")
    exit()

print("[INFO] Memuat model Xception...")
model = tf.keras.models.load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Jalankan Prediksi
print(f"[INFO] Menjalankan prediksi pada {test_generator.samples} frame...")
predictions = model.predict(test_generator, verbose=1)

filenames = test_generator.filenames
true_labels = test_generator.classes

raw_data = []
for i in range(len(filenames)):
    video_name = os.path.basename(filenames[i]).split('_f')[0]
    raw_data.append({
        'video_name': video_name,
        'true_label': true_labels[i],
        'score': float(predictions[i][0])
    })

# Mengelompokkan data per video menggunakan
df_result = pd.DataFrame(raw_data)
video_summary = df_result.groupby('video_name').agg({
    'true_label': 'first',
    'score': 'mean'
}).reset_index()

# Tentukan prediksi akhir (0 atau 1)
video_summary['prediction'] = (video_summary['score'] > 0.5).astype(int)

# TAMPILKAN LAPORAN AKHIR
y_true = video_summary['true_label']
y_pred = video_summary['prediction']

print("\n" + "="*50)
print("HASIL ANALISIS MODEL XCEPTION PADA C40")
print("="*50)
print(f"Total Video Teruji : {len(video_summary)}")
print(f"Akurasi Keseluruhan: {accuracy_score(y_true, y_pred) * 100:.2f}%")
print("-" * 50)
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Original', 'Deepfakes']))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Original', 'Deepfakes'],
            yticklabels=['Original', 'Deepfakes'])
plt.title('Confusion Matrix Xception - C40')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()