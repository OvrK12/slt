import os
import gzip
import gdown
import tarfile
import shutil
import wget
import zipfile

OUTPUT_DIR = './raw'

raw_files = [
    'https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/20110111-annotated-groundtruth.xml.gz',
    'https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/rwth-phoenix-full-20120323.corpus.gz',
    'https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/tracking-groundtruth-sequences.tgz',
    'https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/translation_full_set_20120820.tgz',
    'https://drive.google.com/uc?id=1P7a2nZcD8wFkar8QNCKn_rI0K_I9OHcW' # extracted bodyparts
]

def extract_gz(gz_path, extract_path):
    """This function extract gz-files. 

    Args:
        gz_path: Source path
        extract_path: Target path
    """
    with gzip.open(gz_path, 'rb') as f_in:
        with open(extract_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def extract_tgz(tgz_path, extract_dir):
    """This function extract tgz-files. 

    Args:
        tgz_path: Source path
        extract_dir: Target path
    """
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)

# download files
for url in raw_files:
    if 'google' in url:
        gdown.download(url,output=f'{OUTPUT_DIR}/tracking-groundtruth-sequences-bodyparts.zip')
    else:
        wget.download(url, out=OUTPUT_DIR)

# unpack files
for filename in os.listdir(OUTPUT_DIR):
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if filename.endswith('.gz') and not filename.endswith('.tgz'):
        # Extract .gz files
        extract_path = os.path.join(OUTPUT_DIR, filename[:-3])
        extract_gz(file_path, extract_path)
    
    elif filename.endswith('.zip'):
        # Extract .zip files
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(OUTPUT_DIR)

    elif filename.endswith('.tgz'):
        # Extract .tgz files
        extract_tgz(file_path, OUTPUT_DIR)

    print(f'Extracted: {filename}')