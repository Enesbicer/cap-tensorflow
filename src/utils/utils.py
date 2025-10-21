import os
import cv2
import tensorflow as tf
import urllib
import hashlib
import numpy as np
from sdks.novavision.src.helper.package import PackageHelper

def preprocess_tensorflow(np_image, model):
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    np_image = np_image.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    np_image = (np_image - mean) / std
    
    input_tensor = np.expand_dims(np_image, axis=0)
    
    return input_tensor

def load_storage(storageID):
    result = PackageHelper.get_storage_details(storageID)
    data = result["data"]
    url_path = result["data_url"]
    name = data["name"]
    hash_file = data["hash_file"]
    file_path = f"/storage/{name}"
    storage = os.listdir("/storage")
    if name in storage:
        md5_hash_file = md5_hash(file_path)
        if md5_hash_file != hash_file:
            urllib.request.urlretrieve(url_path, file_path)
    else:
        urllib.request.urlretrieve(url_path, file_path)
    weight_path = f"/storage/{name}"
    return weight_path

def md5_hash(file_path, chunk_size=8192):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as file:
        while True:
            data = file.read(chunk_size)
            if not data:
                break
            md5_hash.update(data)
    return md5_hash.hexdigest()
