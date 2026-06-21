import cv2
import dlib
import numpy as np
import os
import glob
from scipy.spatial import distance as dist

# Dlib
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def calculate_ear(eye):
    # Menghitung jarak vertikal dan horizontal untuk EAR
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def process_dataset(base_path, quality_label):
    categories = ['original', 'Deepfakes']
    main_output_dir = "extracted_features"

    for cat in categories:
        input_dir = os.path.join(base_path, cat)
        output_dir = os.path.join(main_output_dir, quality_label, cat)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        video_files = glob.glob(os.path.join(input_dir, '*.mp4'))


        print(f"MULAI EKSTRAKSI: Kualitas {quality_label} | Kategori {cat.upper()}")
        print(f"Total: {len(video_files)} video")

        for i, video_path in enumerate(video_files, 1):
            video_name = os.path.basename(video_path)
            output_filename = f"{video_name}.npy"
            output_path = os.path.join(output_dir, output_filename)

            if os.path.exists(output_path):
                print(f"[{i}/{len(video_files)}] SKIP: {video_name} (Sudah diproses)")
                continue

            print(f"[{i}/{len(video_files)}] PROCESSING: {video_name}...", end="\r")

            cap = cv2.VideoCapture(video_path)
            video_ear_data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)

                for rect in rects:
                    shape = predictor(gray, rect)
                    coords = np.zeros((68, 2), dtype="int")
                    for j in range(0, 68):
                        coords[j] = (shape.part(j).x, shape.part(j).y)

                    # Ekstraksi koordinat mata dan hitung EAR
                    leftEye = coords[36:42]
                    rightEye = coords[42:48]
                    ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0
                    video_ear_data.append(ear)

            cap.release()

            # Simpan hasil ekstraksi
            if video_ear_data:
                np.save(output_path, np.array(video_ear_data))
                print(f"[{i}/{len(video_files)}] SUCCESS: {video_name} ({len(video_ear_data)} frame)")
            else:
                print(f"[{i}/{len(video_files)}] FAILED: {video_name} (Wajah tidak terdeteksi)")


# Program
path_c23 = './dataset/FaceForensics++_C23'
path_c40 = './dataset/FaceForensics++_C40'

if __name__ == "__main__":
    if os.path.exists(path_c23) and os.path.exists(path_c40):
        process_dataset(path_c23, "C23")
        process_dataset(path_c40, "C40")
        print("SEMUA TAHAPAN EKSTRAKSI SELESAI!")
        print(f"Lokasi Hasil: {os.path.abspath('extracted_features')}")
    else:
        print("Error: Periksa kembali path dataset Anda (C23 & C40).")