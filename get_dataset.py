import requests
import os
import zipfile

def download_dataset(url: str,
                     file: str,
                     dir: str):

    path = f"{file}.zip"

    if os.path.isfile(path):
        print("File already exists. Skipping download.")
    else:
        response = requests.get(url)
        with open(path, "wb") as f:
            f.write(response.content)

    if os.path.isdir(f"{dir}"):
        print("Directory already exists. Skipping extraction.")
    else:
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall()
