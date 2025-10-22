import keras

from keras.applications import (
    Xception,
    VGG16,
    VGG19,
    ResNet50,
    ResNet50V2,
    ResNet101,
    ResNet101V2,
    ResNet152,
    ResNet152V2,
    InceptionV3,
    InceptionResNetV2,
    MobileNet,
    MobileNetV2,
    DenseNet121,
    DenseNet169,
    DenseNet201,
    NASNetMobile,
    NASNetLarge,
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2S,
    EfficientNetV2M,
    EfficientNetV2L,
    ConvNeXtTiny,
    ConvNeXtSmall,
    ConvNeXtBase,
    ConvNeXtLarge,
    ConvNeXtXLarge
)


model_map = {
    # Xception
    "xception": Xception,

    # VGG
    "vgg16": VGG16,
    "vgg19": VGG19,

    # ResNet
    "resnet50": ResNet50,
    "resnet50_v2": ResNet50V2,
    "resnet101": ResNet101,
    "resnet101_v2": ResNet101V2,
    "resnet152": ResNet152,
    "resnet152_v2": ResNet152V2,

    # Inception
    "inception_v3": InceptionV3,
    "inception_resnet_v2": InceptionResNetV2,

    # MobileNet
    "mobilenet": MobileNet,
    "mobilenet_v2": MobileNetV2,

    # DenseNet
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,

    # NASNet
    "nasnet_mobile": NASNetMobile,
    "nasnet_large": NASNetLarge,

    # EfficientNet V1
    "efficientnet_b0": EfficientNetB0,
    "efficientnet_b1": EfficientNetB1,
    "efficientnet_b2": EfficientNetB2,
    "efficientnet_b3": EfficientNetB3,
    "efficientnet_b4": EfficientNetB4,
    "efficientnet_b5": EfficientNetB5,
    "efficientnet_b6": EfficientNetB6,
    "efficientnet_b7": EfficientNetB7,

    # EfficientNet V2
    "efficientnet_v2_b0": EfficientNetV2B0,
    "efficientnet_v2_b1": EfficientNetV2B1,
    "efficientnet_v2_b2": EfficientNetV2B2,
    "efficientnet_v2_b3": EfficientNetV2B3,
    "efficientnet_v2_s": EfficientNetV2S,
    "efficientnet_v2_m": EfficientNetV2M,
    "efficientnet_v2_l": EfficientNetV2L,

    # ConvNeXt
    "convnext_tiny": ConvNeXtTiny,
    "convnext_small": ConvNeXtSmall,
    "convnext_base": ConvNeXtBase,
    "convnext_large": ConvNeXtLarge,
    "convnext_xlarge": ConvNeXtXLarge
}
