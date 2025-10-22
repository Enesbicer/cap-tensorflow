
from sdks.novavision.src.helper.package import PackageHelper
from capsules.Tensorflow.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    ClassificationExecutor,
    ClassificationResponse,
    ClassificationOutputs,
    OutputDetections
)


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
