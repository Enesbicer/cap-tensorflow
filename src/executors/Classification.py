import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from capsules.Tensorflow.src.utils.loads import load_classification_model
from sdks.novavision.src.helper.executor import Executor
from capsules.Tensorflow.src.models.PackageModel import PackageModel
from capsules.Tensorflow.src.utils.response import build_response_classifier
from capsules.Tensorflow.src.utils.prediction import create_predictor


class Classification(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.model = bootstrap.get("model")
        self.model_name = bootstrap.get("model_name")
        self.predictor = bootstrap.get("predictor")
        self.images = self.request.get_param("inputImage")
        self.predictions = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        """Model ve predictor'ı yükler"""
        from sdks.novavision.src.base.logger import LoggerManager
        from sdks.novavision.src.base.application import Application

        # Logger ve application oluştur
        logger = LoggerManager()
        application = Application()

        # Model adını al
        model_name = application.get_param(config=config, name="classificationWeights")

        # Model yükle
        model = load_classification_model(
            application=application,
            config=config,
            logger=logger
        )

        # Predictor oluştur
        predictor = create_predictor(
            application=application,
            config=config,
            logger=logger,
            model=model,
            model_name=model_name
        )

        return {
            "model": model,
            "model_name": model_name,
            "predictor": predictor
        }

    def run(self):
        """Classification işlemini gerçekleştirir"""
        try:
            # Görüntüyü al
            img = Image.get_frame(img=self.images, redis_db=self.redis_db)

            # Prediction yap
            results = self.predictor.predict(img.value)

            # Sonuçları kaydet
            for result in results:
                self.predictions.append({
                    "image_id": img.uID,
                    "class_id": result["class_id"],
                    "confidence": result["confidence"],
                    "confidence_percentage": result["confidence_percentage"]
                })

            # Response oluştur ve döndür
            return build_response_classifier(context=self)

        except Exception as e:
            self.logger.error(f"Classification run hatası: {str(e)}")
            raise


if "__main__" == __name__:
    Executor(sys.argv[1]).run()