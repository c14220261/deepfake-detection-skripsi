import cv2
import os
import glob

# Path dataset C23
base_input = './dataset/FaceForensics++_C23'
# Path target hasil C40
base_output = './dataset/FaceForensics++_C40'

# kategori
categories = ['original', 'Deepfakes']

for cat in categories:
    input_dir = os.path.join(base_input, cat)
    output_dir = os.path.join(base_output, cat)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Mencari semua file .mp4 di dalam kategori original dan deepfakes
    video_files = glob.glob(os.path.join(input_dir, '**/*.mp4'), recursive=True)

    for video_path in video_files:
        file_name = os.path.basename(video_path)
        save_path = os.path.join(output_dir, file_name)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        print(f"[{cat.upper()}] Mengonversi {file_name} ke C40...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
            _, encimg = cv2.imencode('.jpg', frame, encode_param)
            decoded_frame = cv2.imdecode(encimg, 1)
            out.write(decoded_frame)

        cap.release()
        out.release()

print("Proses pemisahan dan konversi C40 selesai!")