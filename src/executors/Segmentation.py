
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from capsules.Tensorflow.src.configs.config import CONFIG
from capsules.Tensorflow.src.utils.loads import ModelLoader
from sdks.novavision.src.helper.executor import Executor
from capsules.Tensorflow.src.models.PackageModel import PackageModel
from capsules.Tensorflow.src.utils.response import build_response_segment
from capsules.Tensorflow.src.classes.TensorflowVision import TensorFlowSegmenter


class Segmentation(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.model = self.bootstrap.get("model")
        self.device = self.bootstrap.get("device")
        self.images = self.request.get_param("inputImage")
        self.conf_threshold = self.request.get_param("conf_threshold")
        self.config_model_type = self.request.get_param("ConfigSegmentationModelType")
        self.model_name = self.request.get_param("SegmentationWeights")
        self.load_parameters()
        self.predictions = []

    def load_parameters(self):
        self.config_type = self.request.get_param("ConfigType")
        if self.config_type == CONFIG["Semantic"]:
            self.model_name = self.request.get_param("SemanticWeights")
        elif self.config_type == CONFIG["Instance"]:
            self.model_name = self.request.get_param("InstanceWeights")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model = ModelLoader(config=config).load_models()
        return model

    def run(self):
        img = Image.get_frame(img=self.images, redis_db=self.redis_db)
        segmenter = TensorFlowSegmenter(self)
        segmenter.predict(img.value, img.uID)
        return  build_response_segment(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
