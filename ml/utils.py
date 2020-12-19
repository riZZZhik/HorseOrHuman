import os
from urllib.request import urlopen
from zipfile import ZipFile


def download_file(url, path="zip.zip"):
    url = urlopen(url)
    with open(path, "wb") as file:
        file.write(url.read())
        file.close()


def unzip_archive(zip_path, folder_path, remove_zip=True):
    zf = ZipFile(zip_path)
    zf.extractall(path=folder_path)
    zf.close()

    if remove_zip:
        os.remove(zip_path)
