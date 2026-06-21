import numpy as np
import os
import glob
import joblib
import pandas as pd
from scipy.stats import skew
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# model
MODEL_HYBRID_PATH = "hybrid_ear_flow_model.pkl"
# Path folder .npy m
DATASET_FEATURES_C40 = r'extracted_features\C40'  # Path folder .npy mata Anda

if not os.path.exists(MODEL_HYBRID_PATH):
    print(f"[ERROR] Model Hybrid {MODEL_HYBRID_PATH} tidak ditemukan! Pastikan file sudah ada di direktori.")
    exit()



# EKSTRAKSI FITUR MATA
def extract_hybrid_features(data_npy):
    """
    Ekstraksi 9 fitur statistik Spasial-Temporal (EAR) dan Motion (Optical Flow).
    Dapat mendeteksi otomatis struktur data 1 kolom (EAR) maupun 2 kolom (EAR + Flow).
    """
    if data_npy.ndim == 1:
        ear_seq = data_npy
        flow_seq = np.zeros_like(ear_seq)  # Padding jika data flow kosong
    else:
        ear_seq = data_npy[:, 0]
        flow_seq = data_npy[:, 1]

    if len(ear_seq) < 5:
        return np.zeros(9)

    # EAR
    mean_ear = np.mean(ear_seq)
    min_ear = np.min(ear_seq)
    std_ear = np.std(ear_seq)
    skew_ear = skew(ear_seq)
    range_ear = np.max(ear_seq) - np.min(ear_seq)

    # Optical Flow
    mean_flow = np.mean(flow_seq)
    max_flow = np.max(flow_seq)
    var_flow = np.var(flow_seq)

    # Nilai Maksimum EAR
    max_ear = np.max(ear_seq)

    return np.array([
        mean_ear, min_ear, std_ear, skew_ear, range_ear,
        mean_flow, max_flow, var_flow, max_ear
    ])



# JALANKAN PREDIKSI PADA DATASET
print("[INFO] MEMULAI EVALUASI MODEL HYBRID MATA (EAR + FLOW)...")

model_hybrid = joblib.load(MODEL_HYBRID_PATH)
results_ear_flow = []
categories = {'original': 0, 'Deepfakes': 1}

for cat_name, label in categories.items():
    folder_path = os.path.join(DATASET_FEATURES_C40, cat_name)
    files = glob.glob(os.path.join(folder_path, "*.npy"))

    if not files:
        print(f"[!] Folder Kategori '{cat_name}' kosong atau tidak ditemukan di: {folder_path}")
        continue

    print(f"Memproses kategori '{cat_name}': {len(files)} file .npy...")

    for f in files:
        try:
            data = np.load(f)
            if data.size > 0:
                # Ambil base nama file sebagai ID identitas video
                video_name = os.path.basename(f).replace('.npy', '').replace('.mp4', '').strip()

                # Ekstraksi fitur dan prediksi probabilitas (Kelas Deepfakes / indeks 1)
                feat = extract_hybrid_features(data).reshape(1, -1)
                prob = model_hybrid.predict_proba(feat)[0][1]

                results_ear_flow.append({
                    'video_name': video_name,
                    'true_label': label,
                    'score_ef': prob
                })
        except Exception as e:
            print(f"Error membaca file {f}: {e}")


# ANALISIS DAN EVALUASI METRIK
df_res = pd.DataFrame(results_ear_flow)

if df_res.empty:
    print("[ERROR] Tidak ada data .npy yang berhasil diproses dan diprediksi.")
    exit()

df_res['prediction_ef'] = (df_res['score_ef'] > 0.5).astype(int)

y_true = df_res['true_label']
y_pred = df_res['prediction_ef']

print(" HASIL EVALUASI MODEL EYE BIOMETRICS (EAR + OPTICAL FLOW)")
print(f"Total Video Teruji : {len(df_res)}")
print(f"Akurasi Keseluruhan: {accuracy_score(y_true, y_pred) * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Original', 'Deepfakes']))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=['Original', 'Deepfakes'],
            yticklabels=['Original', 'Deepfakes'])
plt.title(f'Confusion Matrix EAR + Optical Flow - C40\nAcc: {accuracy_score(y_true, y_pred) * 100:.2f}%')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

print("\n[INFO] Eksekusi testing mandiri EAR + Optical Flow selesai.")