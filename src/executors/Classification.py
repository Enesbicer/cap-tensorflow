import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from capsules.Tensorflow.src.utils.loads import ModelLoader
from sdks.novavision.src.helper.executor import Executor
from capsules.Tensorflow.src.models.PackageModel import PackageModel
from capsules.Tensorflow.src.utils.response import build_response_classifier
from capsules.Tensorflow.src.classes.TensorflowVision import TensorFlowClassifier


class Classification(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.model = self.bootstrap.get("model")
        self.device = self.bootstrap.get("device")
        self.images = self.request.get_param("inputImage")
        self.num_predictions = self.request.get_param("NumPredictions")
        self.model_name = self.request.get_param("ClassificationWeights")
        self.config_model_type = self.request.get_param("ConfigClassificationModelType")
        self.predictions = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model = ModelLoader(config=config).load_models()
        return model

    def run(self):
        img = Image.get_frame(img=self.images, redis_db=self.redis_db)
        classifier = TensorFlowClassifier(self)
        classifier.predict(img.value, img.uID)
        return build_response_classifier(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
