import cv2
import os


# Masukkan path video C23 asli
INPUT_VIDEO_PATH = r"video test original c23/id34_0004.mp4"

# Masukkan path target tempat video hasil kompresi C40 akan disimpan
OUTPUT_VIDEO_PATH = r"video test original c40/id34_0004.mp4"


# PROSES KONVERSI DAN DEGRADASI TEKSTUR C40

def convert_single_video_to_c40(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"[ERROR] Berkas video input '{input_path}' tidak ditemukan!")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Memulai konversi: {os.path.basename(input_path)}")
    print(f"[INFO] Total frame terdeteksi: {total_frames}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mengompresi kualitas piksel frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
        _, encimg = cv2.imencode('.jpg', frame, encode_param)
        decoded_frame = cv2.imdecode(encimg, 1)
        out.write(decoded_frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"       Memproses frame: {frame_count}/{total_frames}...")

    cap.release()
    out.release()

    print(f"[SUKSES] Video C40 berhasil disimpan di: {output_path}\n")

if __name__ == "__main__":
    convert_single_video_to_c40(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)