import cv2
import os
import glob
import sys

def extract_frames_for_xception(base_video_path, output_base_path, frames_per_video=30):
    categories = ['original', 'Deepfakes']
    total_extracted_all = 0

    for cat in categories:
        input_dir = os.path.join(base_video_path, cat)
        save_dir = os.path.join(output_base_path, cat)

        # Membuat folder target jika belum ada
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Mencari video mp4
        video_files = glob.glob(os.path.join(input_dir, '*.mp4'))
        num_videos = len(video_files)

        print(f"\n{'=' * 70}")
        print(f"MULAI EKSTRAKSI FRAME C40: {cat.upper()} ({num_videos} Video)")
        print(f"{'=' * 70}")

        for i, video_path in enumerate(video_files, 1):
            video_name = os.path.basename(video_path)
            last_frame_check = os.path.join(save_dir, f"{video_name}_f{frames_per_video - 1}.jpg")
            if os.path.exists(last_frame_check):
                print(f"[{i}/{num_videos}] SKIP: {video_name} (Sudah diekstrak)")
                continue

            print(f"[{i}/{num_videos}] PROCESSING: {video_name}...", end="\r")

            cap = cv2.VideoCapture(video_path)
            count = 0
            extracted_count = 0

            while cap.isOpened() and extracted_count < frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % 10 == 0:
                    frame_filename = f"{video_name}_f{extracted_count}.jpg"
                    frame_path = os.path.join(save_dir, frame_filename)

                    if not os.path.exists(frame_path):
                        cv2.imwrite(frame_path, frame)

                    extracted_count += 1
                count += 1

            cap.release()
            total_extracted_all += extracted_count

            if extracted_count > 0:
                print(f"[{i}/{num_videos}] SUCCESS: {video_name} ({extracted_count} frame)")
            else:
                print(f"[{i}/{num_videos}] FAILED: {video_name} (Cek file mp4)")

    return total_extracted_all

current_dir = os.path.dirname(os.path.abspath(__file__))
video_path_c40 = os.path.join(current_dir, 'dataset', 'FaceForensics++_C40')
frames_output_path_c40 = os.path.join(current_dir, 'dataset', 'frames_C40')

if __name__ == "__main__":
    if os.path.exists(video_path_c40):
        total = extract_frames_for_xception(video_path_c40, frames_output_path_c40)
        print(f"\n{'=' * 70}")
        print(f"EKSTRAKSI C40 SELESAI!")
        print(f"Total frame baru disimpan: {total}")
        print(f"Lokasi: {os.path.abspath(frames_output_path_c40)}")
        print(f"{'=' * 70}")
    else:
        print(f"Error: Folder dataset C40 tidak ditemukan di: {video_path_c40}")
        print("Pastikan folder 'FaceForensics++_C40' ada di dalam folder 'dataset'.")