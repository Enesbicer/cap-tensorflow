
from sdks.novavision.src.helper.package import PackageHelper
from capsules.Pytorch.src.models.PackageModel import (
    PackageModel, \
    PackageConfigs, \
    ConfigExecutor, \
    DetectionExecutor, ClassificationExecutor, SegmentationExecutor, \
    DetectionResponse, ClassificationResponse, SegmentationResponse, \
    DetectionOutputs, ClassificationOutputs, SegmentationOutputs, \
    OutputDetections
)

def build_response_detector(context):
    outputInfer = OutputDetections(value=context.predictions)
    objectDetectionOutputs = DetectionOutputs(outputDetections=outputInfer)
    objectDetectionResponse = DetectionResponse(outputs=objectDetectionOutputs)
    objectDetectionExecutor = DetectionExecutor(value=objectDetectionResponse)
    executor = ConfigExecutor(value=objectDetectionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_classifier(context):
    outputClassificationResults = OutputDetections(value=context.predictions)
    imageClassificationOutputs = ClassificationOutputs(outputDetections=outputClassificationResults)
    imageClassificationResponse = ClassificationResponse(outputs=imageClassificationOutputs)
    imageClassificationExecutor = ClassificationExecutor(value=imageClassificationResponse)
    executor = ConfigExecutor(value=imageClassificationExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_segment(context):
    outputDetections = OutputDetections(value=context.predictions)
    objectDetectionOutputs = SegmentationOutputs(outputDetections=outputDetections)
    objectDetectionResponse = SegmentationResponse(outputs=objectDetectionOutputs)
    objectDetectionExecutor = SegmentationExecutor(value=objectDetectionResponse)
    executor = ConfigExecutor(value=objectDetectionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel
