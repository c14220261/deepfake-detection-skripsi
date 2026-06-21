import os
import cv2
import dlib
import joblib
import numpy as np
import tensorflow as tf
from scipy.stats import skew
from tensorflow.keras.applications.xception import preprocess_input


# model
MODEL_XENTION_PATH = "deepfake_detector_xception.h5"
MODEL_HYBRID_PATH = "hybrid_ear_flow_model.pkl"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

IMG_SIZE = (299, 299)
BATCH_SIZE = 32
CONFIDENCE_DELTA_THRESHOLD = 0.15

# cek apakah file sudah ada
for path in [MODEL_XENTION_PATH, MODEL_HYBRID_PATH, SHAPE_PREDICTOR_PATH]:
    if not os.path.exists(path):
        print(f"[ERROR] File '{path}' tidak ditemukan!")
        exit()

print("[INFO] Memuat seluruh model ke memori komputasi...")
model_x = tf.keras.models.load_model(MODEL_XENTION_PATH)
model_hybrid = joblib.load(MODEL_HYBRID_PATH)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# FITUR BIOMETRIK MATA
def calculate_ear(eye_points):
    """Menghitung nilai Eye Aspect Ratio (EAR) berdasarkan jarak geometri landmark"""
    a = np.linalg.norm(eye_points[1] - eye_points[5])
    b = np.linalg.norm(eye_points[2] - eye_points[4])
    c = np.linalg.norm(eye_points[0] - eye_points[3])
    return (a + b) / (2.0 * c)


def extract_hybrid_features_from_list(ear_history, flow_history):
    """Mengonstruksi 9 parameter vektor fitur statistik dari historis sekuens video"""
    ear_seq = np.array(ear_history)
    flow_seq = np.array(flow_history)

    if len(ear_seq) < 5:
        return np.zeros(9)

    calculated_skew = skew(ear_seq) if np.std(ear_seq) > 0 else 0.0
    return np.array([
        np.mean(ear_seq),  # Rata-rata EAR
        np.min(ear_seq),  # Nilai Minimum EAR
        np.std(ear_seq),  # Standar Deviasi EAR
        calculated_skew,  # Skewness EAR
        np.max(ear_seq) - np.min(ear_seq),  # Dinamis Kelopak Mata
        np.mean(flow_seq),  # Rata-rata Magnitudo Gerakan
        np.max(flow_seq),  # Lonjakan Maksimum Pergerakan
        np.var(flow_seq),  # Varians Getaran Temporal
        np.max(ear_seq)  # Nilai Maksimum EAR
    ])


# PIPELINE PEMROSESAN VIDEO
def predict_video(video_path, frames_per_video=150):
    if not os.path.exists(video_path):
        print(f"[ERROR] Berkas video '{video_path}' tidak valid.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka file video: {video_path}")
        return

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Memproses video: {os.path.basename(video_path)}")
    print(f"[INFO] Lebar Frame Video Terdeteksi: {video_width} piksel")
    print(f"[INFO] Total frame terdeteksi di dalam berkas: {total_frames}")

    if video_width < 1920:
        CENTER_SHIFT = 0.745
        STEEPNESS = 13.5
        CALIBRATED_THRESHOLD = 0.50
        use_deblocking = True
        print("[AUTOMATED CONTROL] Kualitas Fisik Video Rendah/Terkompresi. Mengaktifkan Deblocking Filter & Parameter C40.")

    else:
        CENTER_SHIFT = 0.94
        STEEPNESS = 20
        CALIBRATED_THRESHOLD = 0.50
        use_deblocking = False
        print("[AUTOMATED CONTROL] Kualitas Fisik Video Ideal (High Resolution). Menggunakan Parameter C23.")

    frames_for_xception = []
    ear_history = []
    flow_history = []
    prev_gray_eye = None
    count = 0
    extracted_count = 0

    while cap.isOpened() and extracted_count < frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break

        if count % 2 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            if len(rects) > 0:
                rect = rects[0]
                shape = predictor(gray, rect)
                points = np.array([[p.x, p.y] for p in shape.parts()])

                h_img, w_img, _ = frame.shape
                x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                x, y = max(0, x), max(0, y)
                w = min(w, w_img - x)
                h = min(h, h_img - y)

                face_img = frame[y:y + h, x:x + w]

                if face_img.size > 0:
                    face_resized = cv2.resize(face_img, IMG_SIZE)

                    if use_deblocking:
                        face_deblocked = cv2.bilateralFilter(face_resized, d=9, sigmaColor=75, sigmaSpace=75)
                        gaussian_blur = cv2.GaussianBlur(face_deblocked, (5, 5), 0)
                        face_resized = cv2.addWeighted(face_deblocked, 1.5, gaussian_blur, -0.5, 0)

                    face_normalized = preprocess_input(face_resized.astype(np.float32))
                    frames_for_xception.append(face_normalized)

                left_eye_pts = points[36:42]
                right_eye_pts = points[42:48]
                ear_left = calculate_ear(left_eye_pts)
                ear_right = calculate_ear(right_eye_pts)
                current_ear = (ear_left + ear_right) / 2.0
                ear_history.append(current_ear)

                eye_all_pts = points[36:48]
                ex, ey, ew, eh = cv2.boundingRect(eye_all_pts)
                ex, ey = max(0, ex), max(0, ey)
                ew = min(ew, w_img - ex)
                eh = min(eh, h_img - ey)
                eye_gray = gray[ey:ey + eh, ex:ex + ew]

                current_flow_magnitude = 0.0
                if prev_gray_eye is not None and prev_gray_eye.shape == eye_gray.shape:
                    if eye_gray.size > 0:
                        flow = cv2.calcOpticalFlowFarneback(prev_gray_eye, eye_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        current_flow_magnitude = np.mean(magnitude)

                flow_history.append(current_flow_magnitude)
                prev_gray_eye = eye_gray
                extracted_count += 1

        count += 1

    cap.release()

    if len(frames_for_xception) == 0 or len(ear_history) < 5:
        print(f"[ERROR] Gagal mengunci elemen wajah pada video ini.")
        return

    print(f"[INFO] Berhasil mengunci dan mengekstrak {len(frames_for_xception)} frame terpilih untuk evaluasi.")

    # SIGMOID
    batch_frames = np.array(frames_for_xception)
    preds_x = model_x.predict(batch_frames, batch_size=BATCH_SIZE, verbose=0)
    raw_score_x = float(np.mean(preds_x))

    # Kalibrasi skor spasial menggunakan fungsi aktivasi Sigmoid Adaptif
    score_x = 1.0 / (1.0 + np.exp(-STEEPNESS * (raw_score_x - CENTER_SHIFT)))

    # Ekstraksi probabilitas dari pilar temporal Biometric Analyzer
    features_vector = extract_hybrid_features_from_list(ear_history, flow_history).reshape(1, -1)
    score_ef = float(model_hybrid.predict_proba(features_vector)[0][1])


    # LOGIKA CONFIDENCE ADAPTIVE GATING
    base_distance = abs(raw_score_x - CENTER_SHIFT)

    is_contradiction_anomaly = (raw_score_x < CENTER_SHIFT and score_ef > 0.70) or \
                               (raw_score_x > CENTER_SHIFT and score_ef < 0.30)

    if is_contradiction_anomaly:
        distance_from_uncertainty = base_distance * 0.01  # Menurunkan paksa ke zona ragu (< 0.15)
    else:
        distance_from_uncertainty = base_distance

    gating_status = ""
    final_score = 0.0

    if distance_from_uncertainty >= CONFIDENCE_DELTA_THRESHOLD:
        final_score = score_x
        gating_status = "High Confidence Gating (Murni Hasil Spasial Xception)"
    else:
        final_score = (0.1 * score_x) + (0.9 * score_ef)
        gating_status = "Low Confidence Gating (Fusi Spasio-Temporal Adaptif)"

    final_decision = "FAKE (Deepfakes)" if final_score > CALIBRATED_THRESHOLD else "REAL (Original)"

    print("HASIL PENGECEKAN")
    print(f" Jalur File Video       : {video_path}")
    print(f" Skor Xception          : {score_x * 100:.2f}%")
    print(f" Probabilitas EAR+Flow  : {score_ef * 100:.2f}%")
    print(f" Jarak Confidence       : {distance_from_uncertainty:.4f} (Ambang Delta: {CONFIDENCE_DELTA_THRESHOLD})")
    print(f" Status Gerbang Logika  : {gating_status}")
    print(f" Final Score (Fusi)     : {final_score:.4f} ({final_score * 100:.2f}%)")
    print(f" KESIMPULAN KLASIFIKASI : STATUS VIDEO ADALAH {final_decision}")


if __name__ == "__main__":
    TARGET_VIDEO = r"video test c40 deepfake/id4_id3_0007 c40.mp4"
    predict_video(TARGET_VIDEO, frames_per_video=150)