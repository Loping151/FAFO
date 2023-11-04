import hashlib
import os
import argparse

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def rename_files_in_directory(directory, extension):
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            file_path = os.path.join(directory, filename)
            sha256_hash = calculate_sha256(file_path)
            new_file_name = sha256_hash + extension
            new_file_path = os.path.join(directory, new_file_name)
            os.rename(file_path, new_file_path)
            print(f"File {filename} is renamed as {new_file_name}")

parser = argparse.ArgumentParser(description="Rename files with their sha256 value to prevent duplicate files")
parser.add_argument("directory", type=str, help="File directory")
parser.add_argument("extension", type=str, help="File extension")

args = parser.parse_args()

rename_files_in_directory(args.directory, args.extension)

# python hash.py /path/to/directory .obj