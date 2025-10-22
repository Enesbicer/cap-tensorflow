# config.py - TensorFlow/Keras Version

import tensorflow as tf
from tensorflow.keras.applications import (
    EfficientNetV2M,
    EfficientNetV2S,
    EfficientNetV2L,
    EfficientNetB7,
    InceptionV3,
    ResNet101,
    ResNet152,
    MobileNetV3Large,
    MobileNetV3Small,
    ConvNeXtTiny,
    ConvNeXtSmall,
    ConvNeXtBase,
    ConvNeXtLarge
)

CONFIG = {
    "storage_path": "/storage/",
    "CustomWeight": "CustomWeight",
    "PreTrained": "PreTrained",
    "Semantic": "Semantic",
    "Instance": "Instance",
    "Detection": "Detection",
    "Classification": "Classification",
    "Segmentation": "Segmentation"
}

# TensorFlow Hub URLs for detection and segmentation models
TF_HUB_MODELS = {
    # Detection models (TensorFlow Hub)
    "ssd_mobilenet_v2": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
    "efficientdet_d0": "https://tfhub.dev/tensorflow/efficientdet/d0/1",
    "efficientdet_d1": "https://tfhub.dev/tensorflow/efficientdet/d1/1",
    "efficientdet_d2": "https://tfhub.dev/tensorflow/efficientdet/d2/1",
    "centernet_resnet50_v1": "https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1",
    "centernet_resnet101_v1": "https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1",
    "faster_rcnn_resnet50_v1": "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1",
    "faster_rcnn_resnet101_v1": "https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1",
    "mask_rcnn_inception_resnet_v2": "https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1",

    # Segmentation models (TensorFlow Hub)
    "deeplabv3_mobilenet_v2": "https://tfhub.dev/tensorflow/deeplabv3_mobilenetv2_cityscapes/1",
    "deeplabv3_resnet101": "https://tfhub.dev/tensorflow/deeplabv3_resnet101_cityscapes/1",
}

model_map = {
    # Classification models with Keras Applications
    "efficientnet_v2_m": (EfficientNetV2M, "imagenet"),
    "efficientnet_v2_s": (EfficientNetV2S, "imagenet"),
    "efficientnet_v2_l": (EfficientNetV2L, "imagenet"),
    "efficientnet_b7": (EfficientNetB7, "imagenet"),
    "inception_v3": (InceptionV3, "imagenet"),
    "resnet101": (ResNet101, "imagenet"),
    "resnet152": (ResNet152, "imagenet"),
    "mobilenet_v3_small": (MobileNetV3Small, "imagenet"),
    "mobilenet_v3_large": (MobileNetV3Large, "imagenet"),
    "convnext_tiny": (ConvNeXtTiny, "imagenet"),
    "convnext_small": (ConvNeXtSmall, "imagenet"),
    "convnext_base": (ConvNeXtBase, "imagenet"),
    "convnext_large": (ConvNeXtLarge, "imagenet"),
}