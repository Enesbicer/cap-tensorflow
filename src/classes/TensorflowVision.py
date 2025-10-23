import tensorflow as tf
import numpy as np

from sdks.novavision.src.base.model import BoundingBox
from capsules.Pytorch.src.models.PackageModel import Detection
from capsules.Pytorch.src.configs.config import CONFIG


class TensorFlowBase:
    def __init__(self, obj: object, is_custom: bool):
        self.obj = obj
        self.model = obj.model
        self.model_name = obj.model_name
        self.model_type_is_custom = is_custom
        self.categories = getattr(obj, "categories", None)
        if self.categories is None and not self.model_type_is_custom:
            print(f"Uyarı: {self.model_name} modeli için 'categories' niteliği bulunamadı.")

    def preprocess(self, img: np.ndarray) -> tf.Tensor:
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        img_batch = tf.expand_dims(img_tensor, axis=0)

        if self.model.dtype == tf.float16:
            img_batch = tf.cast(img_batch, dtype=tf.float16)

        return img_batch

    def return_label(self, label_idx: int) -> str:
        if self.model_type_is_custom or self.categories is None:
            return "None"

        try:
            return self.categories[label_idx]
        except (IndexError, TypeError):
            return "Unknown"


class TensorFlowClassifier(TensorFlowBase):
    def __init__(self, obj: object):
        is_custom = (obj.config_model_type == CONFIG["CustomWeight"])
        super().__init__(obj, is_custom)

    def predict(self, img: np.ndarray, uID: str) -> None:
        img_transformed = self.preprocess(img)

        prediction_batch = self.model(img_transformed, training=False)
        prediction_logits = tf.squeeze(prediction_batch, axis=0)
        prediction_scores = tf.nn.softmax(prediction_logits)

        k = int(self.obj.num_predictions)
        scores_tensor, indices_tensor = tf.nn.top_k(prediction_scores, k=k)

        scores = scores_tensor.numpy()
        indices = indices_tensor.numpy()

        height, width = img.shape[:2]
        bounding_box = BoundingBox(left=0, top=0, width=width, height=height)

        for score, index in zip(scores, indices):
            class_id = int(index)
            confidence_score = float(score)

            detection = Detection(
                boundingBox=bounding_box,
                confidence=confidence_score,
                classId=class_id,
                classLabel=self.return_label(class_id),
                imgUID=uID
            )
            self.obj.predictions.append(detection)