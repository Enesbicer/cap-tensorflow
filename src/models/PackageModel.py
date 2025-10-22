
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import  Package, Inputs, Configs, Outputs, Response, Request, Output, Input, Config,Image, Detection, KeyPoints


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image],Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

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
    name: Literal["configHalfTrue"] = "configHalfTrue"
    value: Literal["True"] = "True"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ConfigHalfFalse(Config):
    name: Literal["configHalfFalse"] = "configHalfFalse"
    value: Literal["False"] = "False"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigHalf(Config):
    """
        It enables half-precision (FP16) inference, which can speed up model inference.
    """
    name: Literal["configHalf"] = "configHalf"
    value: Union[ConfigHalfTrue, ConfigHalfFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Half"


class ConfigDeviceGPU(Config):
    name: Literal["configDeviceGPU"] = "configDeviceGPU"
    configHalf: ConfigHalf
    value: Literal["GPU"] = "GPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU"


class ConfigDeviceCPU(Config):
    name: Literal["configDeviceCPU"] = "configDeviceCPU"
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
    name: Literal["configDevice"] = "configDevice"
    value: Union[ConfigDeviceCPU, ConfigDeviceGPU]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Device"


class StorageSource(Config):
    name: Literal["storageSource"] = "storageSource"
    value: int
    type: Literal["string"] = "string"
    field: Literal["filePicker"] = "filePicker"

    class Config:
        json_schema_extra = {
            "class": "portalium\storage\widgets\FilePicker",
            "options": {
                "multiple": 0,
                "returnAttribute": [
                    "name"
                ],
                "name": "app::logo_wide"
            }
        }
        title = "Storage Source"


class StorageId(Config):
    name: Literal["storageId"] = "storageId"
    storageSource: StorageSource
    value: Literal["storageid"] = "storageid"
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Storage ID"


class ConfigConfidentThreshold(Config):
    name: Literal["configConfidentThreshold"] = "configConfidentThreshold"
    value: float = Field(default=0.3, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Confident Threshold"


#Classification

class ConfigWeightsTop5(Config):
    name: Literal["configWeightsTop5"] = "configWeightsTop5"
    value: Literal["5"] = "5"
    type: Literal["number"] = "number"
    field: Literal["option"] = "option"

    class Config:
        title = "Top 5"


class ConfigWeightsTop1(Config):
    name: Literal["configWeightsTop1"] = "configWeightsTop1"
    value: Literal["1"] = "1"
    type: Literal["number"] = "number"
    field: Literal["option"] = "option"

    class Config:
        title = "Top 1"


class ConfigNumPredictions(Config):
    name: Literal["configNumPredictions"] = "configNumPredictions"
    value: Union[ConfigWeightsTop1,ConfigWeightsTop5]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Num Predictions"


class Efficientnet_v2_m(Config):
    name: Literal["efficientnet_v2_m"] = "efficientnet_v2_m"
    value: Literal["efficientnet_v2_m"] = "efficientnet_v2_m"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "efficientnet_v2_m"


class Efficientnet_v2_s(Config):
    name: Literal["efficientnet_v2_s"] = "efficientnet_v2_s"
    value: Literal["efficientnet_v2_s"] = "efficientnet_v2_s"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "efficientnet_v2_s"


class Efficientnet_v2_l(Config):
    name: Literal["efficientnet_v2_l"] = "efficientnet_v2_l"
    value: Literal["efficientnet_v2_l"] = "efficientnet_v2_l"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "efficientnet_v2_l"

class Efficientnet_b7(Config):
    name: Literal["efficientnet_b7"] = "efficientnet_b7"
    value: Literal["efficientnet_b7"] = "efficientnet_b7"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "efficientnet_b7"

class Inception_v3(Config):
    name: Literal["inception_v3"] = "inception_v3"
    value: Literal["inception_v3"] = "inception_v3"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "inception_v3"


class Resnet101(Config):
    name: Literal["resnet101"] = "resnet101"
    value: Literal["resnet101"] = "resnet101"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "resnet101"


class Resnet152(Config):
    name: Literal["resnet152"] = "resnet152"
    value: Literal["resnet152"] = "resnet152"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "resnet152"


class Mobilenet_v3_small(Config):
    name: Literal["mobilenet_v3_small"] = "mobilenet_v3_small"
    value: Literal["mobilenet_v3_small"] = "mobilenet_v3_small"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "mobilenet_v3_small"


class Mobilenet_v3_large(Config):
    name: Literal["mobilenet_v3_large"] = "mobilenet_v3_large"
    value: Literal["mobilenet_v3_large"] = "mobilenet_v3_large"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "mobilenet_v3_large"


class Convnext_tiny(Config):
    name: Literal["convnext_tiny"] = "convnext_tiny"
    value: Literal["convnext_tiny"] = "convnext_tiny"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "convnext_tiny"


class Convnext_small(Config):
    name: Literal["convnext_small"] = "convnext_small"
    value: Literal["convnext_small"] = "convnext_small"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "convnext_small"


class Convnext_base(Config):
    name: Literal["convnext_base"] = "convnext_base"
    value: Literal["convnext_base"] = "convnext_base"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "convnext_base"


class Convnext_large(Config):
    name: Literal["convnext_large"] = "convnext_large"
    value: Literal["convnext_large"] = "convnext_large"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "convnext_large"


class ClassificationWeights(Config):
    name: Literal["classificationWeights"] = "classificationWeights"
    value: Union[Efficientnet_v2_m, Efficientnet_v2_s, Efficientnet_v2_l, Efficientnet_b7, Inception_v3, Resnet101, Resnet152, Mobilenet_v3_small, Mobilenet_v3_large, Convnext_tiny, Convnext_small, Convnext_base, Convnext_large]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Model"


class ClassificationPreTrained(Config):
    classificationWeights: ClassificationWeights
    configDevice: ConfigDevice
    configNumPredictions: ConfigNumPredictions
    name: Literal["preTrained"] = "preTrained"
    value: Literal["PreTrained"] = "PreTrained"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "PreTrained"


class NumClass(Config):
    """
        Specifies the number of classes the model should predict. For example number of labels in your dataset.
    """
    name: Literal["numClass"] = "numClass"
    value: int = Field(ge=0, default=1000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    restart: Literal[True] = True

    class Config:
        title = "Number of Classes"


class ClassificationCustomWeight(Config):
    classificationWeights: ClassificationWeights
    numclass:NumClass
    storageId: StorageId
    configDevice: ConfigDevice
    configNumPredictions: ConfigNumPredictions
    name: Literal["customWeight"] = "customWeight"
    value: Literal["CustomWeight"] = "CustomWeight"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Custom Weight"


class ConfigClassificationModelType(Config):
    name: Literal["configClassificationModelType"] = "configClassificationModelType"
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


class Classification(Config):
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


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[Classification]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["Pytorch"] = "Tensorflow"