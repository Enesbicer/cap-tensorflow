import os
import hashlib
import urllib.request
import tensorflow as tf
import numpy as np
from typing import Tuple
from sdks.novavision.src.helper.package import PackageHelper


def md5_hash(file_path, chunk_size=8192):
    """Dosyanın MD5 hash'ini hesaplar"""
    md5_hash_obj = hashlib.md5()
    with open(file_path, "rb") as file:
        while True:
            data = file.read(chunk_size)
            if not data:
                break
            md5_hash_obj.update(data)
    return md5_hash_obj.hexdigest()


def load_storage(storage_id):
    """Storage'dan model dosyasını indirir"""
    result = PackageHelper.get_storage_details(storage_id)
    data = result["data"]
    url_path = result["data_url"]
    name = data["name"]
    hash_file = data["hash_file"]

    file_path = f"/storage/{name}"

    # Storage dizini yoksa oluştur
    os.makedirs("/storage", exist_ok=True)
    storage = os.listdir("/storage")

    if name in storage:
        md5_hash_file = md5_hash(file_path)
        if md5_hash_file != hash_file:
            urllib.request.urlretrieve(url_path, file_path)
    else:
        urllib.request.urlretrieve(url_path, file_path)

    return file_path


# Model input size mapping
MODEL_INPUT_SIZES = {
    "inception_v3": (299, 299),
    "inception_resnet_v2": (299, 299),
    "xception": (299, 299),
    "nasnet_large": (331, 331),
}


def get_input_size(model_name: str) -> Tuple[int, int]:
    """Model için doğru input size döndürür"""
    return MODEL_INPUT_SIZES.get(model_name.lower(), (224, 224))


def preprocess_image(image: np.ndarray, model_name: str) -> np.ndarray:
    """Görüntüyü TensorFlow modeli için hazırlar"""
    # BGR ise RGB'ye çevir (OpenCV BGR kullanır)
    if image.shape[-1] == 3:
        image = image[..., ::-1]  # BGR -> RGB

    # Grayscale ise RGB yap
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:  # RGBA
        image = image[:, :, :3]

    # Model'e göre resize
    target_size = get_input_size(model_name)
    image = tf.image.resize(image, target_size).numpy()

    # Model-specific preprocessing
    preprocessing_map = {
        "xception": tf.keras.applications.xception.preprocess_input,
        "vgg16": tf.keras.applications.vgg16.preprocess_input,
        "vgg19": tf.keras.applications.vgg19.preprocess_input,
        "resnet50": tf.keras.applications.resnet50.preprocess_input,
        "resnet50_v2": tf.keras.applications.resnet50.preprocess_input,
        "resnet101": tf.keras.applications.resnet.preprocess_input,
        "resnet101_v2": tf.keras.applications.resnet.preprocess_input,
        "resnet152": tf.keras.applications.resnet.preprocess_input,
        "resnet152_v2": tf.keras.applications.resnet.preprocess_input,
        "inception_v3": tf.keras.applications.inception_v3.preprocess_input,
        "inception_resnet_v2": tf.keras.applications.inception_resnet_v2.preprocess_input,
        "mobilenet": tf.keras.applications.mobilenet.preprocess_input,
        "mobilenet_v2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "densenet121": tf.keras.applications.densenet.preprocess_input,
        "densenet169": tf.keras.applications.densenet.preprocess_input,
        "densenet201": tf.keras.applications.densenet.preprocess_input,
        "nasnet_mobile": tf.keras.applications.nasnet.preprocess_input,
        "nasnet_large": tf.keras.applications.nasnet.preprocess_input,
        "efficientnet_b0": tf.keras.applications.efficientnet.preprocess_input,
        "efficientnet_b1": tf.keras.applications.efficientnet.preprocess_input,
        "efficientnet_b2": tf.keras.applications.efficientnet.preprocess_input,
        "efficientnet_b3": tf.keras.applications.efficientnet.preprocess_input,
        "efficientnet_b4": tf.keras.applications.efficientnet.preprocess_input,
        "efficientnet_b5": tf.keras.applications.efficientnet.preprocess_input,
        "efficientnet_b6": tf.keras.applications.efficientnet.preprocess_input,
        "efficientnet_b7": tf.keras.applications.efficientnet.preprocess_input,
        "efficientnet_v2_b0": tf.keras.applications.efficientnet_v2.preprocess_input,
        "efficientnet_v2_b1": tf.keras.applications.efficientnet_v2.preprocess_input,
        "efficientnet_v2_b2": tf.keras.applications.efficientnet_v2.preprocess_input,
        "efficientnet_v2_b3": tf.keras.applications.efficientnet_v2.preprocess_input,
        "efficientnet_v2_s": tf.keras.applications.efficientnet_v2.preprocess_input,
        "efficientnet_v2_m": tf.keras.applications.efficientnet_v2.preprocess_input,
        "efficientnet_v2_l": tf.keras.applications.efficientnet_v2.preprocess_input,
        "convnext_tiny": tf.keras.applications.convnext.preprocess_input,
        "convnext_small": tf.keras.applications.convnext.preprocess_input,
        "convnext_base": tf.keras.applications.convnext.preprocess_input,
        "convnext_large": tf.keras.applications.convnext.preprocess_input,
        "convnext_xlarge": tf.keras.applications.convnext.preprocess_input,
    }

    preprocess_fn = preprocessing_map.get(model_name.lower())

    if preprocess_fn:
        # Batch dimension ekle, preprocess, sonra çıkar
        image = np.expand_dims(image, axis=0)
        image = preprocess_fn(image)
    else:
        # Default: [-1, 1] normalization
        image = (image / 127.5) - 1.0
        image = np.expand_dims(image, axis=0)

    return image