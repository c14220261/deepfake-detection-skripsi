import numpy as np
import os
import glob
import joblib
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# dataset
PATH_DATA_C23 = r'extracted_features\C23'
PATH_DATA_C40 = r'extracted_features\C40'


def extract_stealth_features(ear_sequence):
    """
    Ekstraksi 9 parameter vektor fitur statistik dari sekuens biner EAR.
    Menghitung statistik geometri kelopak mata serta dinamika gerakan temporal
    (Implicit Optical Flow) untuk mendeteksi anomali getaran (jitter).
    """
    if len(ear_sequence) < 5:
        return np.zeros(9)

    # Geometri mata
    mean_v = np.mean(ear_sequence)
    std_v = np.std(ear_sequence)
    min_v = np.min(ear_sequence)
    max_v = np.max(ear_sequence)

    # Intensitas rata-rata gerakan mata
    diff_v = np.diff(ear_sequence)
    velocity_mean = np.mean(np.abs(diff_v))
    velocity_var = np.var(diff_v)

    # Frekuensi Kedipan
    skew_v = skew(ear_sequence) if np.std(ear_sequence) > 0 else 0.0
    kurt_v = kurtosis(ear_sequence) if np.std(ear_sequence) > 0 else 0.0

    # Rentang Pembukaan Kelopak Mata
    range_v = max_v - min_v

    return np.array([
        mean_v, min_v, std_v, skew_v, range_v,
        velocity_mean, max_v, velocity_var, max_v
    ])


def train_model():
    X, y = [], []
    categories = {'original': 0, 'Deepfakes': 1}

    # menggabungkan data C23 dan C40 ke dalam satu matriks training
    for path_domain, domain_name in [(PATH_DATA_C23, "C23"), (PATH_DATA_C40, "C40")]:
        print(f"\n--- Memuat Dataset Domain {domain_name} ---")

        for cat_name, label in categories.items():
            folder = os.path.join(path_domain, cat_name)

            if not os.path.exists(folder):
                print(f"[PREVIEW] Folder {folder} tidak ditemukan. Dilewati...")
                continue

            files = glob.glob(os.path.join(folder, "*.npy"))
            print(f"[INFO] Memproses {len(files)} file .npy dari {domain_name}/{cat_name}")

            for f in files:
                data = np.load(f)
                if data.size > 10:
                    X.append(extract_stealth_features(data))
                    y.append(label)

    if len(X) == 0:
        print("[ERROR] Tidak ada data fitur .npy yang berhasil dimuat. Periksa folder target!")
        return

    print(f"\n[INFO] Total Sampel Gabungan Lintas Domain: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # max_depth=12 & min_samples_leaf=4 membatasi pohon agar tidak menghafal noise.
    # class_weight memberi penalti 1.5x lebih berat jika salah mendeteksi video REAL (0),
    # sehingga probabilitas arah FAKE (1) untuk video asli ditekan agar tetap rendah.
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=12,
        min_samples_leaf=4,
        class_weight={0: 1.5, 1: 1.0}
    )

    print("[INFO] Melatih model Random Forest Classifier...")
    model.fit(X_train, y_train)

    # hasil evaluasi
    y_pred = model.predict(X_test)
    print(f" [HASIL] Akurasi Validasi Gabungan (C23+C40): {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nLaporan Klasifikasi Matriks Performa:")
    print(classification_report(y_test, y_pred, target_names=['Original', 'Deepfakes']))
    output_filename = "hybrid_ear_flow_model.pkl"
    joblib.dump(model, output_filename)
    print(f"[SUKSES] Model robust '{output_filename}' berhasil diperbarui dan disimpan.")


if __name__ == "__main__":
    train_model()