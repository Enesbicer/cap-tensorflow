import tensorflow as tf
import numpy as np
from imantics import Mask

from sdks.novavision.src.base.model import BoundingBox
from capsules.Tensorflow.src.configs.config import CONFIG
from capsules.Tensorflow.src.configs.config import model_map
from capsules.Tensorflow.src.utils.utils import preprocess_tensorflow
from capsules.Tensorflow.src.models.PackageModel import Detection, KeyPoints


class TensorFlowBase:
    def __init__(self, obj: object, model_map: dict, is_custom: bool):
        self.obj = obj
        self.model = obj.model
        self.device = obj.device
        self.model_name = obj.model_name
        self.model_type_is_custom = is_custom
        _, self.weights = model_map[self.model_name]
        self.categories = None if self.model_type_is_custom else self.weights.meta["categories"]

    def preprocess(self, img: np.ndarray):
        img_transformed = preprocess_tensorflow(img, self.model)
        return img_transformed

    def return_label(self, label_idx: int) -> str:
        if self.model_type_is_custom:
            return "None"
        return self.categories[label_idx]


class TensorFlowDetector(TensorFlowBase):
    def __init__(self, obj: object):
        super().__init__(obj, model_map, obj.config_model_type == CONFIG["CustomWeight"])

    def predict(self, img: np.ndarray, uID) -> None:
        img_transformed = self.preprocess(img)
        prediction = self.model(img_transformed, training=False)

        boxes = prediction["boxes"][0]
        labels = prediction["labels"][0]
        scores = prediction["scores"][0]

        for i in range(len(boxes)):
            if scores[i].numpy() > self.obj.conf_threshold:
                x1, y1, x2, y2 = boxes[i].numpy().tolist()
                detection = Detection(
                    boundingBox=BoundingBox(
                        left=x1,
                        top=y1,
                        width=x2 - x1,
                        height=y2 - y1
                    ),
                    confidence=float(scores[i].numpy()),
                    classId=int(labels[i].numpy()),
                    classLabel=self.return_label(int(labels[i].numpy())),
                    imgUID=uID
                )
                self.obj.predictions.append(detection)


class TensorFlowClassifier(TensorFlowBase):
    def __init__(self, obj: object):
        super().__init__(obj, model_map, obj.config_model_type == CONFIG["CustomWeight"])

    def predict(self, img: np.ndarray, uID: str) -> None:
        img_transformed = self.preprocess(img)
        prediction = self.model(img_transformed, training=False)
        prediction = tf.nn.softmax(prediction[0])

        top_k = tf.nn.top_k(prediction, k=int(self.obj.num_predictions))
        scores = top_k.values.numpy()
        indices = top_k.indices.numpy()

        height, width = img.shape[:2]
        bounding_box = BoundingBox(left=0, top=0, width=width, height=height)

        for score, index in zip(scores, indices):
            class_id = int(index)
            detection = Detection(
                boundingBox=bounding_box,
                confidence=float(prediction[class_id].numpy()),
                classId=class_id,
                classLabel=self.return_label(class_id),
                imgUID=uID
            )
            self.obj.predictions.append(detection)


class TensorFlowSegmenter(TensorFlowBase):
    def __init__(self, obj: object):
        super().__init__(obj, model_map, obj.config_model_type == CONFIG["CustomWeight"])

    def predict(self, img: np.ndarray, uID: str) -> None:
        input_tensor = self.preprocess(img)

        if self.obj.config_type == CONFIG["Semantic"]:
            self._predict_semantic(input_tensor, img, uID)
        elif self.obj.config_type == CONFIG["Instance"]:
            self._predict_instance(input_tensor, uID)
        else:
            raise ValueError("config_type must be either Semantic or Instance")

    def _predict_semantic(self, input_tensor, img, uID):
        original_h, original_w = img.shape[:2]
        prediction = self.model(input_tensor, training=False)
        normalized_masks = tf.nn.softmax(prediction, axis=-1)

        for class_id in range(1, normalized_masks.shape[-1]):
            mask = normalized_masks[0, :, :, class_id] > 0.5
            if tf.reduce_sum(tf.cast(mask, tf.int32)) == 0:
                continue
            mask = tf.cast(mask, tf.uint8).numpy()
            polygons = Mask(mask).polygons()
            confidence_values = normalized_masks[0,
                polygons.points[0][:, 1].astype(int),
                polygons.points[0][:, 0].astype(int),
                class_id
            ]

            h, w = mask.shape[-2:]
            keypoints = [
                KeyPoints(cx=int(x * original_w / w), cy=int(y * original_h / h), confidence=float(conf.numpy()))
                for (x, y), conf in zip(polygons.points[0], confidence_values)
            ]

            avg_conf = float(tf.reduce_mean(confidence_values).numpy())
            if avg_conf > self.obj.conf_threshold:
                self.obj.predictions.append(
                    Detection(
                        boundingBox=BoundingBox(left=0, top=0, width=img.shape[1], height=img.shape[0]),
                        keyPoints=keypoints,
                        confidence=round(avg_conf, 2),
                        classId=int(class_id),
                        classLabel=self.return_label(class_id),
                        imgUID=uID,
                        segment_type=self.obj.config_type
                    )
                )

    def _predict_instance(self, input_tensor, uID):
        predictions = self.model(input_tensor, training=False)

        if isinstance(predictions, list):
            predictions = predictions[0]

        boxes = predictions["boxes"][0]
        labels = predictions["labels"][0]
        scores = predictions["scores"][0]
        masks = predictions["masks"][0]

        for i in range(len(boxes)):
            if scores[i] < self.obj.conf_threshold:
                continue

            x1, y1, x2, y2 = boxes[i].numpy().tolist()
            mask = (masks[i, :, :, 0].numpy() > 0.5).astype(np.uint8) * 255
            polygons = Mask(mask).polygons()
            keypoints = [KeyPoints(cx=int(x), cy=int(y)) for x, y in polygons.points[0]]

            self.obj.predictions.append(
                Detection(
                    boundingBox=BoundingBox(
                        left=int(x1),
                        top=int(y1),
                        width=int(x2 - x1),
                        height=int(y2 - y1)
                    ),
                    keyPoints=keypoints,
                    confidence=round(float(scores[i].numpy()), 2),
                    classId=int(labels[i].numpy()),
                    classLabel=self.return_label(int(labels[i].numpy())),
                    imgUID=uID,
                    segment_type=self.obj.config_type
                )
            )