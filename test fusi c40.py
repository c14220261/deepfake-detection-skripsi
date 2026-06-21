import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
import glob
import joblib
import re
from scipy.stats import skew
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_XCEPTION_PATH = "deepfake_detector_xception.h5"
MODEL_HYBRID_PATH = "hybrid_ear_flow_model.pkl"

DATASET_FRAMES_C40 = r'dataset\frames_C40'
DATASET_FEATURES_C40 = r'extracted_features\C40'

IMG_SIZE = (299, 299)
BATCH_SIZE = 32

# Ambang batas deteksi Gating Adaptive
CONFIDENCE_DELTA_THRESHOLD = 0.15

if not os.path.exists(MODEL_XCEPTION_PATH) or not os.path.exists(MODEL_HYBRID_PATH):
    print("[ERROR] Pastikan kedua file model .h5 dan .pkl berada di direktori."); exit()

def clean_video_id(name_string):
    """Penyelarasan ID sesuai dengan bab Desain Data Skripsi"""
    name_cleaned = name_string.replace('.npy', '').replace('.mp4', '').strip()
    match = re.search(r'\d+', name_cleaned)
    return match.group(0) if match else name_cleaned

# SPATIAL ANALYZER (XCEPTION)
print("\n" + "=" * 50)
print("[STEP 1] JALANKAN PILAR 1: SPATIAL ANALYZER (XCEPTION)...")
print("=" * 50)
model_x = tf.keras.models.load_model(MODEL_XCEPTION_PATH)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    DATASET_FRAMES_C40,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"[INFO] Memproses {test_generator.samples} frame...")
predictions_x = model_x.predict(test_generator, verbose=1)
filenames_x = test_generator.filenames
true_labels_x = test_generator.classes

raw_data_x = []
for i in range(len(filenames_x)):
    video_name = os.path.basename(filenames_x[i]).split('_f')[0]
    raw_data_x.append({
        'video_name': clean_video_id(video_name),
        'true_label': true_labels_x[i],
        'score_x': float(predictions_x[i][0])
    })

df_x_summary = pd.DataFrame(raw_data_x).groupby(['video_name', 'true_label']).agg({'score_x': 'mean'}).reset_index()
print(f"[INFO] Prediksi Xception selesai. Ditemukan {len(df_x_summary)} entitas video.")

# BIOMETRIC ANALYZER (EAR + OPTICAL FLOW)
print("[STEP 2] JALANKAN PILAR 2: BIOMETRIC ANALYZER (EAR + FLOW)...")
model_hybrid = joblib.load(MODEL_HYBRID_PATH)

def extract_hybrid_features(data_npy):
    if data_npy.ndim == 1:
        ear_seq = data_npy; flow_seq = np.zeros_like(ear_seq)
    else:
        ear_seq = data_npy[:, 0]; flow_seq = data_npy[:, 1]
    if len(ear_seq) < 5: return np.zeros(9)

    return np.array([
        np.mean(ear_seq), np.min(ear_seq), np.std(ear_seq), skew(ear_seq),
        np.max(ear_seq) - np.min(ear_seq), np.mean(flow_seq), np.max(flow_seq),
        np.var(flow_seq), np.max(ear_seq)
    ])

results_ear_flow = []
categories = {'original': 0, 'Deepfakes': 1}

for cat_name, label in categories.items():
    folder_path = os.path.join(DATASET_FEATURES_C40, cat_name)
    files = glob.glob(os.path.join(folder_path, "*.npy"))
    print(f"[INFO] Memproses {len(files)} file .npy kategori {cat_name}...")
    for f in files:
        try:
            data = np.load(f)
            if data.size > 0:
                video_name = os.path.basename(f).split('_f')[0]
                feat = extract_hybrid_features(data).reshape(1, -1)
                prob = model_hybrid.predict_proba(feat)[0][1]
                results_ear_flow.append({
                    'video_name': clean_video_id(video_name),
                    'true_label': label,
                    'score_ef': prob
                })
        except Exception as e:
            pass

df_res_ef = pd.DataFrame(results_ear_flow)
print(f"[INFO] Prediksi Hybrid EAR+Flow selesai. Ditemukan {len(df_res_ef)} entitas video.")

# CONFIDENCE-BASED ADAPTIVE GATING
print("\n" + "=" * 50)
print("[STEP 3] IMPLEMENTASI MEKANISME ADAPTIVE GATING...")
print("=" * 50)

df_fusion = pd.merge(df_x_summary, df_res_ef, on=['video_name', 'true_label'])
print(f"[INFO] Berhasil mensinkronisasi {len(df_fusion)} video secara valid.")

df_fusion['pred_x'] = (df_fusion['score_x'] > 0.5).astype(int)
df_fusion['pred_ef'] = (df_fusion['score_ef'] > 0.5).astype(int)

final_predictions = []
gating_decisions = []

for idx, row in df_fusion.iterrows():
    s_x = row['score_x']
    s_ef = row['score_ef']

    distance_from_uncertainty = abs(s_x - 0.5)

    if distance_from_uncertainty >= CONFIDENCE_DELTA_THRESHOLD:
        final_score = s_x
        gating_decisions.append('High Confidence (Xception)')
    else:
        final_score = (0.1 * s_x) + (0.9 * s_ef)
        gating_decisions.append('Low Confidence (Biometric Override)')

    final_predictions.append(1 if final_score > 0.5 else 0)

df_fusion['final_prediction'] = final_predictions
df_fusion['gating_status'] = gating_decisions

# OUTPUT
y_true = df_fusion['true_label']

print(" 1. HASIL EVALUASI PILAR 1: SPASIAL ANALYZER (XCEPTION)")
print(f"Akurasi Xception: {accuracy_score(y_true, df_fusion['pred_x']) * 100:.2f}%")
print(classification_report(y_true, df_fusion['pred_x'], target_names=['Original', 'Deepfakes']))

print(" 2. HASIL EVALUASI PILAR 2: BIOMETRIC ANALYZER (EAR + FLOW)")
print(f"Akurasi EAR + Flow: {accuracy_score(y_true, df_fusion['pred_ef']) * 100:.2f}%")
print(classification_report(y_true, df_fusion['pred_ef'], target_names=['Original', 'Deepfakes']))

print(" 3. HASIL EVALUASI SISTEM AKHIR: ADAPTIVE GATING FUSI")
acc_fusi = accuracy_score(y_true, df_fusion['final_prediction']) * 100
print(f"Akurasi Sistem Akhir: {acc_fusi:.2f}%")
print(classification_report(y_true, df_fusion['final_prediction'], target_names=['Original', 'Deepfakes']))

print("RINGKASAN LOGIKA MEKANISME GATING PADA DATA C40:")
print(df_fusion['gating_status'].value_counts())

# Render Confusion Matrix Berdampingan
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.heatmap(confusion_matrix(y_true, df_fusion['pred_x']), annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['Orig', 'Fake'], yticklabels=['Orig', 'Fake'])
axes[0].set_title(f"1. CM Spatial (Xception)\nAcc: {accuracy_score(y_true, df_fusion['pred_x']) * 100:.2f}%")

sns.heatmap(confusion_matrix(y_true, df_fusion['pred_ef']), annot=True, fmt='d', cmap='Oranges', ax=axes[1], xticklabels=['Orig', 'Fake'], yticklabels=['Orig', 'Fake'])
axes[1].set_title(f"2. CM Biometric (EAR+Flow)\nAcc: {accuracy_score(y_true, df_fusion['pred_ef']) * 100:.2f}%")

sns.heatmap(confusion_matrix(y_true, df_fusion['final_prediction']), annot=True, fmt='d', cmap='Purples', ax=axes[2], xticklabels=['Orig', 'Fake'], yticklabels=['Orig', 'Fake'])
axes[2].set_title(f"3. CM Adaptive Gating\nAcc: {acc_fusi:.2f}%")
plt.tight_layout()
plt.show()