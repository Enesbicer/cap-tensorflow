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
    "Classification": "Classification"
}

class WeightsConfig:
    def __init__(self, weights='imagenet', meta=None):
        self.weights = weights
        self.meta = meta if meta else {}

imagenet_categories = [f"class_{i}" for i in range(1000)]

model_map = {
    "efficientnet_v2_m": (EfficientNetV2M, WeightsConfig('imagenet', {'categories': imagenet_categories})),
    "efficientnet_v2_s": (EfficientNetV2S, WeightsConfig('imagenet', {'categories': imagenet_categories})),
    "efficientnet_v2_l": (EfficientNetV2L, WeightsConfig('imagenet', {'categories': imagenet_categories})),
    "efficientnet_b7": (EfficientNetB7, WeightsConfig('imagenet', {'categories': imagenet_categories})),
    "inception_v3": (InceptionV3, WeightsConfig('imagenet', {'categories': imagenet_categories})),
    "resnet101": (ResNet101, WeightsConfig('imagenet', {'categories': imagenet_categories})),
    "resnet152": (ResNet152, WeightsConfig('imagenet', {'categories': imagenet_categories})),
    "mobilenet_v3_small": (MobileNetV3Small, WeightsConfig('imagenet', {'categories': imagenet_categories})),
    "mobilenet_v3_large": (MobileNetV3Large, WeightsConfig('imagenet', {'categories': imagenet_categories})),
    "convnext_tiny": (ConvNeXtTiny, WeightsConfig('imagenet', {'categories': imagenet_categories})),
    "convnext_small": (ConvNeXtSmall, WeightsConfig('imagenet', {'categories': imagenet_categories})),
    "convnext_base": (ConvNeXtBase, WeightsConfig('imagenet', {'categories': imagenet_categories})),
    "convnext_large": (ConvNeXtLarge, WeightsConfig('imagenet', {'categories': imagenet_categories}))
}
