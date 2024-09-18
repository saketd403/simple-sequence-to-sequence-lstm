import os
import requests
import zipfile
from pathlib import Path


def create_dir(file_list):

    for file in file_list:
        if file.is_dir():
            print(f"{file} directory exists.")
        else:
            print(f"Did not find {file} directory, creating one...")
            file.mkdir(parents=True, exist_ok=True)

def download_files(file_list,download_ls):

    for file, download_path in zip(file_list,download_ls):

        file_name = download_path.split("/")[-1]
        with open(file / file_name, "wb") as f:
            request = requests.get(download_path)
            f.write(request.content)

        if(file_name.endswith(".zip")):
            with zipfile.ZipFile(file / file_name, "r") as zip_ref:
                zip_ref.extractall(file)

#Get current directory
current_directory = Path.cwd()

# Setup path to data folder
file_path_en = current_directory / "data" / "word_embeddings_en"
file_path_de = current_directory / "data" / "word_embeddings_de"

file_path_ls = [file_path_en,file_path_de]
download_path_ls = ["https://nlp.stanford.edu/data/glove.6B.zip","https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt"]

create_dir(file_path_ls)

download_files(file_path_ls,download_path_ls)






