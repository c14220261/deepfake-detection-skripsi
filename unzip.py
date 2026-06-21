import zipfile
import os

zip_file_path = 'ff-c23.zip'

extraction_path = './dataset'

if not os.path.exists(extraction_path):
    os.makedirs(extraction_path)

print("Memulai ekstraksi file...")

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

print(f"Ekstraksi selesai! File tersimpan di: {extraction_path}")
print("Isi folder dataset:", os.listdir(extraction_path))
