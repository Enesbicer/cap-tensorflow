from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Inputs, Configs, Outputs, Response, Request, Output, Input, Config,Image, Detection, KeyPoints


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        val = values.get('value')
        if isinstance(val, Image):
            return "object"
        elif isinstance(val, list):
            return "list"
        return "object"

    class Config:
        title = "Image"


class KeyPoints(KeyPoints):
    confidence: Optional[float] = 0.0


class Detection(Detection):
    keyPoints: Optional[List[KeyPoints]] = []
    index: Optional[int] = None
    imgUID: Optional[str] = ""
    segmentType: Optional[str] = ""


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class ConfigHalfTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ConfigHalfFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigHalf(Config):
    """
        It enables half-precision (FP16) inference, which can speed up model inference.
    """
    name: Literal["Half"] = "Half"
    value: Union[ConfigHalfTrue, ConfigHalfFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Half"


class ConfigDeviceGPU(Config):
    name: Literal["ConfigDeviceGPU"] = "ConfigDeviceGPU"
    configHalf: ConfigHalf
    value: Literal["GPU"] = "GPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU"


class ConfigDeviceCPU(Config):
    name: Literal["ConfigDeviceCPU"] = "ConfigDeviceCPU"
    value: Literal["CPU"] = "CPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CPU"


class ConfigDevice(Config):
    """
        It refers to whether the model should run on a CPU or a GPU.
        You can select the device type for inference or training process.
    """
    name: Literal["ConfigDevice"] = "ConfigDevice"
    value: Union[ConfigDeviceCPU, ConfigDeviceGPU]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Device"


class CustomFieldStorageId(Config):
    name: Literal["Id"] = "Id"
    value: int
    type: Literal["string"] = "string"
    field: Literal["filePicker"] = "filePicker"

    class Config:
        json_schema_extra = {
            "class": "portalium\\storage\\widgets\\FilePicker",
            "options": {
                "multiple": 0,
                "returnAttribute": [
                    "name"
                ],
                "name": "app::logo_wide"
            }
        }
        title = "Storage Source"


class CustomFieldStorage(Config):
    name: Literal["storageid"] = "storageid"
    storageID: CustomFieldStorageId
    value: Literal["storageid"] = "storageid"
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Storage ID"


class ConfigConfidentThreshold(Config):
    name: Literal["conf_threshold"] = "conf_threshold"
    value: float = Field(default=0.3, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Confident Threshold"




class DetectionWeights1(Config):
    name: Literal["ssd_mobilenet_v2"] = "ssd_mobilenet_v2"
    value: Literal["ssd_mobilenet_v2"] = "ssd_mobilenet_v2"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "ssd_mobilenet_v2"


class DetectionWeights2(Config):
    name: Literal["efficientdet_d0"] = "efficientdet_d0"
    value: Literal["efficientdet_d0"] = "efficientdet_d0"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "efficientdet_d0"


class DetectionWeights3(Config):
    name: Literal["efficientdet_d1"] = "efficientdet_d1"
    value: Literal["efficientdet_d1"] = "efficientdet_d1"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "efficientdet_d1"


class DetectionWeights4(Config):
    name: Literal["efficientdet_d2"] = "efficientdet_d2"
    value: Literal["efficientdet_d2"] = "efficientdet_d2"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "efficientdet_d2"


class DetectionWeights5(Config):
    name: Literal["centernet_resnet50_v1"] = "centernet_resnet50_v1"
    value: Literal["centernet_resnet50_v1"] = "centernet_resnet50_v1"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "centernet_resnet50_v1_fpn_512x512"


class DetectionWeights6(Config):
    name: Literal["centernet_resnet101_v1"] = "centernet_resnet101_v1"
    value: Literal["centernet_resnet101_v1"] = "centernet_resnet101_v1"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "centernet_resnet101_v1_fpn_512x512"


class DetectionWeights7(Config):
    name: Literal["faster_rcnn_resnet50_v1"] = "faster_rcnn_resnet50_v1"
    value: Literal["faster_rcnn_resnet50_v1"] = "faster_rcnn_resnet50_v1"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "faster_rcnn_resnet50_v1_640x640"


class DetectionWeights8(Config):
    name: Literal["faster_rcnn_resnet101_v1"] = "faster_rcnn_resnet101_v1"
    value: Literal["faster_rcnn_resnet101_v1"] = "faster_rcnn_resnet101_v1"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "faster_rcnn_resnet101_v1_640x640"


class DetectionWeights9(Config):
    name: Literal["mask_rcnn_inception_resnet_v2"] = "mask_rcnn_inception_resnet_v2"
    value: Literal["mask_rcnn_inception_resnet_v2"] = "mask_rcnn_inception_resnet_v2"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "mask_rcnn_inception_resnet_v2_1024x1024"


class DetectionWeights(Config):
    name: Literal["DetectionWeights"] = "DetectionWeights"
    value: Union[
        DetectionWeights1,
        DetectionWeights2,
        DetectionWeights3,
        DetectionWeights4,
        DetectionWeights5,
        DetectionWeights6,
        DetectionWeights7,
        DetectionWeights8,
        DetectionWeights9
    ]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Model"


class DetectionPreTrained(Config):
    detectionWeights: DetectionWeights
    configDevice: ConfigDevice
    configConfidentThreshold: ConfigConfidentThreshold
    name: Literal["PreTrained"] = "PreTrained"
    value: Literal["PreTrained"] = "PreTrained"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "PreTrained"


class DetectionCustomWeight(Config):
    detectionWeights: DetectionWeights
    customFieldStorage: CustomFieldStorage
    configDevice: ConfigDevice
    configConfidentThreshold: ConfigConfidentThreshold
    name: Literal["CustomWeight"] = "CustomWeight"
    value: Literal["CustomWeight"] = "CustomWeight"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Custom Weight"


class ConfigDetectionModelType(Config):
    name: Literal["ConfigDetectionModelType"] = "ConfigDetectionModelType"
    value: Union[DetectionPreTrained, DetectionCustomWeight]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Model Type"


class DetectionInputs(Inputs):
    inputImage: InputImage


class DetectionConfigs(Configs):
    configDetectionModelType: ConfigDetectionModelType


class DetectionOutputs(Outputs):
    outputDetections: OutputDetections


class DetectionRequest(Request):
    inputs: Optional[DetectionInputs]
    configs: DetectionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class DetectionResponse(Response):
    outputs: DetectionOutputs


class DetectionExecutor(Config):
    name: Literal["Detection"] = "Detection"
    value: Union[DetectionRequest, DetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }




class ConfigWeightsTop5(Config):
    name: Literal["top5"] = "top5"
    value: Literal["5"] = "5"
    type: Literal["number"] = "number"
    field: Literal["option"] = "option"

    class Config:
        title = "Top 5"


class ConfigWeightsTop1(Config):
    name: Literal["top1"] = "top1"
    value: Literal["1"] = "1"
    type: Literal["number"] = "number"
    field: Literal["option"] = "option"

    class Config:
        title = "Top 1"


class ConfigNumPredictions(Config):
    name: Literal["NumPredictions"] = "NumPredictions"
    value: Union[ConfigWeightsTop1, ConfigWeightsTop5]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Num Predictions"


class ClassificationWeights1(Config):
    name: Literal["efficientnet_v2_m"] = "efficientnet_v2_m"
    value: Literal["efficientnet_v2_m"] = "efficientnet_v2_m"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "efficientnet_v2_m"


class ClassificationWeights2(Config):
    name: Literal["efficientnet_v2_s"] = "efficientnet_v2_s"
    value: Literal["efficientnet_v2_s"] = "efficientnet_v2_s"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "efficientnet_v2_s"


class ClassificationWeights3(Config):
    name: Literal["efficientnet_v2_l"] = "efficientnet_v2_l"
    value: Literal["efficientnet_v2_l"] = "efficientnet_v2_l"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "efficientnet_v2_l"


class ClassificationWeights4(Config):
    name: Literal["efficientnet_b7"] = "efficientnet_b7"
    value: Literal["efficientnet_b7"] = "efficientnet_b7"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "efficientnet_b7"


class ClassificationWeights5(Config):
    name: Literal["inception_v3"] = "inception_v3"
    value: Literal["inception_v3"] = "inception_v3"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "inception_v3"


class ClassificationWeights6(Config):
    name: Literal["resnet101"] = "resnet101"
    value: Literal["resnet101"] = "resnet101"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "resnet101"


class ClassificationWeights7(Config):
    name: Literal["resnet152"] = "resnet152"
    value: Literal["resnet152"] = "resnet152"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "resnet152"


class ClassificationWeights8(Config):
    name: Literal["mobilenet_v3_small"] = "mobilenet_v3_small"
    value: Literal["mobilenet_v3_small"] = "mobilenet_v3_small"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "mobilenet_v3_small"


class ClassificationWeights9(Config):
    name: Literal["mobilenet_v3_large"] = "mobilenet_v3_large"
    value: Literal["mobilenet_v3_large"] = "mobilenet_v3_large"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "mobilenet_v3_large"


class ClassificationWeights10(Config):
    name: Literal["convnext_tiny"] = "convnext_tiny"
    value: Literal["convnext_tiny"] = "convnext_tiny"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "convnext_tiny"


class ClassificationWeights11(Config):
    name: Literal["convnext_small"] = "convnext_small"
    value: Literal["convnext_small"] = "convnext_small"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "convnext_small"


class ClassificationWeights12(Config):
    name: Literal["convnext_base"] = "convnext_base"
    value: Literal["convnext_base"] = "convnext_base"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "convnext_base"


class ClassificationWeights13(Config):
    name: Literal["convnext_large"] = "convnext_large"
    value: Literal["convnext_large"] = "convnext_large"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "convnext_large"


class ClassificationWeights(Config):
    name: Literal["ClassificationWeights"] = "ClassificationWeights"
    value: Union[
        ClassificationWeights1,
        ClassificationWeights2,
        ClassificationWeights3,
        ClassificationWeights4,
        ClassificationWeights5,
        ClassificationWeights6,
        ClassificationWeights7,
        ClassificationWeights8,
        ClassificationWeights9,
        ClassificationWeights10,
        ClassificationWeights11,
        ClassificationWeights12,
        ClassificationWeights13
    ]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Model"


class ClassificationPreTrained(Config):
    classificationWeights: ClassificationWeights
    configDevice: ConfigDevice
    configNumPredictions: ConfigNumPredictions
    name: Literal["PreTrained"] = "PreTrained"
    value: Literal["PreTrained"] = "PreTrained"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "PreTrained"


class NumClass(Config):
    """
        Specifies the number of classes the model should predict. For example number of labels in your dataset.
    """
    name: Literal["NumClass"] = "NumClass"
    value: int = Field(ge=0, default=1000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    restart: Literal[True] = True

    class Config:
        title = "Number of Classes"


class ClassificationCustomWeight(Config):
    classificationWeights: ClassificationWeights
    numclass: NumClass
    customFieldStorage: CustomFieldStorage
    configDevice: ConfigDevice
    configNumPredictions: ConfigNumPredictions
    name: Literal["CustomWeight"] = "CustomWeight"
    value: Literal["CustomWeight"] = "CustomWeight"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Custom Weight"


class ConfigClassificationModelType(Config):
    name: Literal["ConfigClassificationModelType"] = "ConfigClassificationModelType"
    value: Union[ClassificationPreTrained, ClassificationCustomWeight]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Model Type"


class ClassificationInputs(Inputs):
    inputImage: InputImage


class ClassificationConfigs(Configs):
    configClassificationModelType: ConfigClassificationModelType


class ClassificationOutputs(Outputs):
    outputDetections: OutputDetections


class ClassificationRequest(Request):
    inputs: Optional[ClassificationInputs]
    configs: ClassificationConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ClassificationResponse(Response):
    outputs: ClassificationOutputs


class ClassificationExecutor(Config):
    name: Literal["Classification"] = "Classification"
    value: Union[ClassificationRequest, ClassificationResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Classification"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }




class SegmentationWeights1(Config):
    name: Literal["hrnet_ade20k_w48"] = "hrnet_ade20k_w48"
    value: Literal["hrnet_ade20k_w48"] = "hrnet_ade20k_w48"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "hrnet_ade20k_w48"


class SegmentationWeights2(Config):
    name: Literal["mobile_food_segmenter_v1"] = "mobile_food_segmenter_v1"
    value: Literal["mobile_food_segmenter_v1"] = "mobile_food_segmenter_v1"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "mobile_food_segmenter_v1"


class SegmentationWeights3(Config):
    name: Literal["deeplabv3_mobilenetv2_ade20k"] = "deeplabv3_mobilenetv2_ade20k"
    value: Literal["deeplabv3_mobilenetv2_ade20k"] = "deeplabv3_mobilenetv2_ade20k"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "deeplabv3_mobilenetv2_ade20k"


class SegmentationWeights(Config):
    name: Literal["SegmentationWeights"] = "SegmentationWeights"
    value: Union[SegmentationWeights1, SegmentationWeights2, SegmentationWeights3]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Weights"




class SegmentationPreTrained(Config):
    segmentationWeights: SegmentationWeights  # configType yerine geldi
    configDevice: ConfigDevice
    configConfidentThreshold: ConfigConfidentThreshold
    name: Literal["PreTrained"] = "PreTrained"
    value: Literal["PreTrained"] = "PreTrained"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "PreTrained"


class SegmentationCustomWeight(Config):
    segmentationWeights: SegmentationWeights  # configType yerine geldi
    customFieldStorage: CustomFieldStorage
    configDevice: ConfigDevice
    configConfidentThreshold: ConfigConfidentThreshold
    name: Literal["CustomWeight"] = "CustomWeight"
    value: Literal["CustomWeight"] = "CustomWeight"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Custom Weight"


class ConfigSegmentationModelType(Config):
    name: Literal["ConfigSegmentationModelType"] = "ConfigSegmentationModelType"
    value: Union[SegmentationPreTrained, SegmentationCustomWeight]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Model Type"


class SegmentationInputs(Inputs):
    inputImage: InputImage


class SegmentationConfigs(Configs):
    configSegmentationModelType: ConfigSegmentationModelType


class SegmentationOutputs(Outputs):
    outputDetections: OutputDetections


class SegmentationRequest(Request):
    inputs: Optional[SegmentationInputs]
    configs: SegmentationConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class SegmentationResponse(Response):
    outputs: SegmentationOutputs


class SegmentationExecutor(Config):
    name: Literal["Segmentation"] = "Segmentation"
    value: Union[SegmentationRequest, SegmentationResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Segmentation"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }




class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[DetectionExecutor, ClassificationExecutor, SegmentationExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["Tensorflow"] = "Tensorflow"