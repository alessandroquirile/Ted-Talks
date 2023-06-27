import os
import zipfile

def extract(folder_path: str):
    if os.path.exists(folder_path):
        extract_directory = "./"
        with zipfile.ZipFile(folder_path, 'r') as zip_ref:
            zip_ref.extractall(extract_directory)
        print(folder_path, "extracted")
    else:
        print(folder_path, "does not exist")

extract("ted_talks_it.zip")
