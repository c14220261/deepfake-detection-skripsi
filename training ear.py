import numpy as np
import os
import glob
import joblib
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

PATH_DATA = r'extracted_features\C40'


def extract_hybrid_features(data_npy):
    if data_npy.ndim == 1:
        ear_seq = data_npy
        flow_seq = np.zeros_like(ear_seq) 
    else:
        ear_seq = data_npy[:, 0]
        flow_seq = data_npy[:, 1]

    if len(ear_seq) < 5: return np.zeros(9)

    # EAR
    mean_ear = np.mean(ear_seq)
    min_ear = np.min(ear_seq)
    std_ear = np.std(ear_seq)
    skew_ear = skew(ear_seq)
    range_ear = np.max(ear_seq) - np.min(ear_seq)

    # Flow
    mean_flow = np.mean(flow_seq)
    max_flow = np.max(flow_seq)
    var_flow = np.var(flow_seq)

    return np.array([
        mean_ear, min_ear, std_ear, skew_ear, range_ear,
        mean_flow, max_flow, var_flow, np.max(ear_seq)
    ])


def train_hybrid():
    X, y = [], []
    categories = {'original': 0, 'Deepfakes': 1}

    print(f"[INFO] Mencari data di: {os.path.abspath(PATH_DATA)}")

    for cat_name, label in categories.items():
        folder = os.path.join(PATH_DATA, cat_name)
        files = glob.glob(os.path.join(folder, "*.npy"))

        if len(files) == 0:
            print(f"[PERINGATAN] Tidak ada file .npy ditemukan di: {folder}")
            continue

        print(f"[INFO] Memproses {len(files)} file dari {cat_name}...")
        for f in files:
            try:
                data = np.load(f)
                if data.size > 0:
                    X.append(extract_hybrid_features(data))
                    y.append(label)
            except Exception as e:
                print(f"Gagal memuat {f}: {e}")

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("\n[ERROR] Data training kosong! Periksa path folder Anda.")
        print(f"Path yang dicari: {PATH_DATA}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Melatih model pada {len(X_train)} sampel...")
    rf = RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("\n" + "=" * 50)
    print(f"AKURASI MODEL: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("=" * 50)

    joblib.dump(rf, "hybrid_ear_flow_model.pkl")
    print("[SUKSES] Model disimpan sebagai: hybrid_ear_flow_model.pkl")


if __name__ == "__main__":
    train_hybrid()
