import os
import cv2
import urllib
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sdks.novavision.src.helper.package import PackageHelper


def preprocess_tensorflow(np_image, model_name):
    """
    Tensorflow/Keras modelleri için görüntü ön işleme

    Args:
        np_image: BGR formatında numpy array
        model_name: Kullanılacak model ismi (configs.py'deki key)

    Returns:
        Preprocessed tensor ready for inference
    """
    # BGR'den RGB'ye çevir
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    # Model-specific preprocessing function'ı al
    from keras.applications import (
        xception, vgg16, vgg19, resnet, resnet_v2,
        inception_v3, inception_resnet_v2,
        mobilenet, mobilenet_v2,
        densenet, nasnet,
        efficientnet, efficientnet_v2,
        convnext
    )

    # Model ailesine göre doğru preprocessing fonksiyonunu seç
    preprocess_map = {
        # Xception
        "xception": xception.preprocess_input,

        # VGG
        "vgg16": vgg16.preprocess_input,
        "vgg19": vgg19.preprocess_input,

        # ResNet V1
        "resnet50": resnet.preprocess_input,
        "resnet101": resnet.preprocess_input,
        "resnet152": resnet.preprocess_input,

        # ResNet V2
        "resnet50_v2": resnet_v2.preprocess_input,
        "resnet101_v2": resnet_v2.preprocess_input,
        "resnet152_v2": resnet_v2.preprocess_input,

        # Inception
        "inception_v3": inception_v3.preprocess_input,
        "inception_resnet_v2": inception_resnet_v2.preprocess_input,

        # MobileNet
        "mobilenet": mobilenet.preprocess_input,
        "mobilenet_v2": mobilenet_v2.preprocess_input,

        # DenseNet
        "densenet121": densenet.preprocess_input,
        "densenet169": densenet.preprocess_input,
        "densenet201": densenet.preprocess_input,

        # NASNet
        "nasnet_mobile": nasnet.preprocess_input,
        "nasnet_large": nasnet.preprocess_input,

        # EfficientNet V1
        "efficientnet_b0": efficientnet.preprocess_input,
        "efficientnet_b1": efficientnet.preprocess_input,
        "efficientnet_b2": efficientnet.preprocess_input,
        "efficientnet_b3": efficientnet.preprocess_input,
        "efficientnet_b4": efficientnet.preprocess_input,
        "efficientnet_b5": efficientnet.preprocess_input,
        "efficientnet_b6": efficientnet.preprocess_input,
        "efficientnet_b7": efficientnet.preprocess_input,

        # EfficientNet V2
        "efficientnet_v2_b0": efficientnet_v2.preprocess_input,
        "efficientnet_v2_b1": efficientnet_v2.preprocess_input,
        "efficientnet_v2_b2": efficientnet_v2.preprocess_input,
        "efficientnet_v2_b3": efficientnet_v2.preprocess_input,
        "efficientnet_v2_s": efficientnet_v2.preprocess_input,
        "efficientnet_v2_m": efficientnet_v2.preprocess_input,
        "efficientnet_v2_l": efficientnet_v2.preprocess_input,

        # ConvNeXt
        "convnext_tiny": convnext.preprocess_input,
        "convnext_small": convnext.preprocess_input,
        "convnext_base": convnext.preprocess_input,
        "convnext_large": convnext.preprocess_input,
        "convnext_xlarge": convnext.preprocess_input,
    }

    # Model-specific preprocessing uygula
    preprocess_fn = preprocess_map.get(model_name, keras.applications.imagenet_utils.preprocess_input)

    # Batch dimension ekle ve preprocess
    np_image = np.expand_dims(np_image, axis=0)
    preprocessed = preprocess_fn(np_image)

    return preprocessed


def load_storage(storageID):
    """
    Storage'dan model ağırlıklarını indir ve yükle
    Hash kontrolü ile güncelliği doğrula

    Args:
        storageID: Storage ID from API

    Returns:
        weight_path: Model weight dosyasının yolu
    """
    result = PackageHelper.get_storage_details(storageID)
    data = result["data"]
    url_path = result["data_url"]
    name = data["name"]
    hash_file = data["hash_file"]

    file_path = f"/storage/{name}"
    storage = os.listdir("/storage")

    # Dosya varsa ve hash eşleşmiyorsa yeniden indir
    if name in storage:
        md5_hash_file = md5_hash(file_path)
        if md5_hash_file != hash_file:
            urllib.request.urlretrieve(url_path, file_path)
    else:
        # Dosya yoksa indir
        urllib.request.urlretrieve(url_path, file_path)

    weight_path = f"/storage/{name}"
    return weight_path


def md5_hash(file_path, chunk_size=8192):
    """
    Dosyanın MD5 hash'ini hesapla

    Args:
        file_path: Dosya yolu
        chunk_size: Okuma chunk boyutu

    Returns:
        MD5 hash string
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as file:
        while True:
            data = file.read(chunk_size)
            if not data:
                break
            md5_hash.update(data)
    return md5_hash.hexdigest()