import os
import gzip
import tarfile
import shutil
import wget


OUTPUT_DIR = './raw'

raw_files = [
    'https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/20110111-annotated-groundtruth.xml.gz',
    'https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/rwth-phoenix-full-20120323.corpus.gz',
    'https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/tracking-groundtruth-sequences.tgz',
    'https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/translation_full_set_20120820.tgz'
]

def extract_gz(gz_path, extract_path):
    with gzip.open(gz_path, 'rb') as f_in:
        with open(extract_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def extract_tgz(tgz_path, extract_dir):
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)

# download files
for url in raw_files:
    wget.download(url, out=OUTPUT_DIR)

# unpack files
for filename in os.listdir(OUTPUT_DIR):
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if filename.endswith('.gz') and not filename.endswith('.tgz'):
        # Extract .gz files
        extract_path = os.path.join(OUTPUT_DIR, filename[:-3])
        extract_gz(file_path, extract_path)
        print(f"Extracted: {filename}")
    
    elif filename.endswith('.tgz'):
        # Extract .tgz files
        extract_tgz(file_path, OUTPUT_DIR)
        print(f"Extracted: {filename}")