import numpy as np
import tensorflow as tf
from tensorflow import keras
from sdks.novavision.src.base.model import BoundingBox
from capsules.Tensorflow.src.configs.configs import model_map
from capsules.Tensorflow.src.utils.utils import preprocess_tensorflow
from capsules.Tensorflow.src.models.PackageModel import Detection


class KerasClassifier:
    def __init__(self, obj: object):
        self.obj = obj
        self.model = obj.model
        self.device = obj.device
        self.model_name = obj.model_name
        self.model_type_is_custom = obj.config_model_type == "CustomWeight"

        # Model metadata'dan kategorileri al
        self.categories = self._get_categories()

    def _get_categories(self):
        """
        Model kategorilerini al
        Custom model için None, ImageNet için categories listesi döner
        """
        if self.model_type_is_custom:
            return None

        # ImageNet categories (1000 sınıf)
        # Keras ImageNet decode_predictions fonksiyonu kullanılabilir
        try:
            from keras.applications.imagenet_utils import decode_predictions
            return "imagenet"  # decode_predictions kullanılacak
        except ImportError:
            return None

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Görüntüyü model için hazırla
        Model-specific preprocessing uygula
        """
        preprocessed = preprocess_tensorflow(img, self.model_name)

        # Mixed precision kontrolü
        if self.device == "GPU":
            policy = tf.keras.mixed_precision.global_policy()
            if policy.compute_dtype == 'float16':
                preprocessed = tf.cast(preprocessed, tf.float16)

        return preprocessed

    def return_label(self, label_idx: int, prediction_array: np.ndarray = None) -> str:
        """
        Class ID'den label döndür

        Args:
            label_idx: Class index
            prediction_array: Tüm prediction array (ImageNet decode için)

        Returns:
            Class label string
        """
        if self.model_type_is_custom:
            return f"class_{label_idx}"

        # ImageNet modeller için decode_predictions kullan
        if self.categories == "imagenet" and prediction_array is not None:
            from keras.applications.imagenet_utils import decode_predictions
            # decode_predictions batch için çalışır, tek prediction için expand dims
            decoded = decode_predictions(prediction_array, top=1000)
            # label_idx'e karşılık gelen label'ı bul
            for decoded_prediction in decoded[0]:
                class_id, class_name, _ = decoded_prediction
                if class_id == label_idx or decoded[0].index(decoded_prediction) == label_idx:
                    return class_name
            return f"class_{label_idx}"

        return f"class_{label_idx}"

    def predict(self, img: np.ndarray, uID: str) -> None:
        """
        Classification prediction yap ve sonuçları obj.predictions'a ekle

        Args:
            img: Input image (BGR numpy array)
            uID: Unique image identifier
        """
        # Preprocessing
        img_transformed = self.preprocess(img)

        # Inference
        with tf.device('/GPU:0' if self.device == "GPU" else '/CPU:0'):
            prediction = self.model.predict(img_transformed, verbose=0)

        # Softmax uygula (model output'unda yoksa)
        # Keras modeller genelde son layer'da softmax içerir ama kontrol edelim
        if prediction.max() > 1.0:  # Softmax uygulanmamış
            prediction = tf.nn.softmax(prediction, axis=-1).numpy()

        # Top-K prediction al
        top_k = int(self.obj.num_predictions)

        # NumPy ile top-k indices ve scores
        top_indices = np.argsort(prediction[0])[-top_k:][::-1]
        top_scores = prediction[0][top_indices]

        # Full image bounding box oluştur
        height, width = img.shape[:2]
        bounding_box = BoundingBox(left=0, top=0, width=width, height=height)

        # Her top prediction için Detection objesi oluştur
        for idx, score in zip(top_indices, top_scores):
            class_id = int(idx)
            confidence = float(score)

            # Confidence threshold kontrolü (opsiyonel)
            if hasattr(self.obj, 'conf_threshold') and confidence < self.obj.conf_threshold:
                continue

            detection = Detection(
                boundingBox=bounding_box,
                confidence=confidence,
                classId=class_id,
                classLabel=self.return_label(class_id, prediction),
                imgUID=uID
            )

            self.obj.predictions.append(detection)


# Geriye dönük uyumluluk için alias
TensorflowClassifier = KerasClassifier