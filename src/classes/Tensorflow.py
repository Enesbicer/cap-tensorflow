import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from imantics import Mask
import tf_keras  # tf.keras olarak güncellendi (Eğer TF 2.x+ kullanılıyorsa)

# --- Orijinal kodunuzdaki bu importların aynı olduğunu varsayıyoruz ---
from sdks.novavision.src.base.model import BoundingBox
from capsules.Tensorflow.src.models.PackageModel import Detection, KeyPoints

# CONFIG ve model_map'in TF için yeniden düzenlenmesi gerekiyor
# -----------------------------------------------------------------

# ÖRNEK CONFIG ve Model Haritası (TensorFlow için)
# Gerçek uygulamada, bu haritanın çok daha kapsamlı olması gerekir.
CONFIG = {
    "CustomWeight": "custom",
    "Semantic": "semantic",
    "Instance": "instance"
}

# TensorFlow model_map eşdeğeri
# Keras App'ler için sınıfın kendisini, TF Hub için URL'yi saklayabiliriz.
tf_model_map = {
    # Sınıflandırma
    "resnet50_keras": (tf_keras.applications.ResNet50, tf_keras.applications.resnet.preprocess_input, "imagenet"),

    # Nesne Tespiti (TF Hub'dan)
    "faster_rcnn_resnet50_tfhub": (
        "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1",
        None,  # Ön işleme modelin içinde veya basittir
        "coco"  # Kategori setinin adı (manuel olarak eşlenmeli)
    ),

    # Instance Segmentasyon (TF Hub'dan)
    "mask_rcnn_inception_resnet_tfhub": (
        "https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1",
        None,
        "coco"
    ),

    # Semantic Segmentasyon (TF Hub'dan)
    "deeplab_v3_tfhub": (
        "https://tfhub.dev/tensorflow/deeplabv3/mnv2_dm05_coco/1/default/1",
        (513, 513),  # Bu model özel bir giriş boyutu bekler
        "coco"
    )
}

# COCO kategorileri (TF Hub modelleri genellikle 1-indekslidir)
# Bu listeyi manuel olarak sağlamamız gerekir.
COCO_CATEGORIES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    # ... (90'a kadar devam eder)
}


class TensorFlowBase:
    def __init__(self, obj: object, model_map: dict, is_custom: bool):
        self.obj = obj
        self.model_name = obj.model_name
        self.model_type_is_custom = is_custom

        model_entry, self.preprocess_fn, self.category_set_name = model_map[self.model_name]

        if self.model_type_is_custom:
            # Özel ağırlıklar, tf.keras.models.load_model ile yüklenir
            self.model = tf_keras.models.load_model(obj.model_path)
            self.categories = None
        elif isinstance(model_entry, str):
            # TensorFlow Hub modeli (URL'den yüklenir)
            self.model = hub.load(model_entry)
            self.categories = COCO_CATEGORIES if self.category_set_name == "coco" else None
        else:
            # tf.keras.applications modeli
            self.model = model_entry(weights=self.category_set_name)
            self.categories = COCO_CATEGORIES if self.category_set_name == "coco" else None  # ImageNet için ayrı bir map gerekir

    def preprocess(self, img: np.ndarray):
        """
        Görüntüyü TensorFlow modeline hazırlar.
        """
        # Görüntüyü bir batch'e dönüştür (Batch size = 1)
        img_batch = np.expand_dims(img, axis=0)

        if self.preprocess_fn is None:
            # TF Hub modelleri genellikle uint8 veya [0,1] float bekler
            # Model belgelerine bakılmalıdır.
            # Örn: Detection modelleri genelde uint8 ister.
            return tf.convert_to_tensor(img_batch, dtype=tf.uint8)

        elif isinstance(self.preprocess_fn, tuple):
            # Semantic segmentasyon modeli gibi özel boyutlandırma
            resize_h, resize_w = self.preprocess_fn
            img_resized = tf.image.resize(img_batch, (resize_h, resize_w))
            # Genellikle [0, 1] arası normalizasyon
            return tf.cast(img_resized, tf.float32) / 255.0

        else:
            # Keras applications (örn: resnet.preprocess_input)
            return self.preprocess_fn(img_batch)

    def return_label(self, label_idx: int) -> str:
        if self.model_type_is_custom or self.categories is None:
            return "None"
        # get(label_idx, "Unknown") -> etiketi bulamazsa "Unknown" döner
        return self.categories.get(int(label_idx), "Unknown")


class TensorFlowDetector(TensorFlowBase):
    def __init__(self, obj: object):
        super().__init__(obj, tf_model_map, obj.config_model_type == CONFIG["CustomWeight"])

    def predict(self, img: np.ndarray, uID) -> None:
        img_transformed = self.preprocess(img)

        # TF Hub modelleri genellikle bir dict döndürür
        # Keras modelleri .predict() kullanır, TF Hub modelleri doğrudan çağrılır
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(img_transformed)  # Özel Keras modeli
            # ... (Özel modelin çıktısını burada işlemeniz gerekir)
        else:
            prediction = self.model(img_transformed)  # TF Hub modeli

        # TF Hub çıktılarını NumPy'a dönüştür (EagerTensor'dan)
        boxes = prediction["detection_boxes"][0].numpy()
        labels = prediction["detection_classes"][0].numpy()
        scores = prediction["detection_scores"][0].numpy()

        original_h, original_w = img.shape[:2]

        for i in range(len(boxes)):
            if scores[i] > self.obj.conf_threshold:
                # --- KOORDİNAT DÖNÜŞÜMÜ (ÇOK ÖNEMLİ) ---
                # TF Hub: [ymin, xmin, ymax, xmax] (Normalize)
                # Bizim İhtiyacımız: [x1, y1, x2, y2] (Piksel)
                ymin, xmin, ymax, xmax = boxes[i]

                x1 = xmin * original_w
                y1 = ymin * original_h
                x2 = xmax * original_w
                y2 = ymax * original_h

                width = x2 - x1
                height = y2 - y1
                # --- Dönüşüm Sonu ---

                class_id = int(labels[i])
                detection = Detection(
                    boundingBox=BoundingBox(
                        left=x1,
                        top=y1,
                        width=width,
                        height=height
                    ),
                    confidence=scores[i].item(),
                    classId=class_id,
                    classLabel=self.return_label(class_id),
                    imgUID=uID
                )
                self.obj.predictions.append(detection)


class TensorFlowClassifier(TensorFlowBase):
    def __init__(self, obj: object):
        super().__init__(obj, tf_model_map, obj.config_model_type == CONFIG["CustomWeight"])

    def predict(self, img: np.ndarray, uID: str) -> None:
        img_transformed = self.preprocess(img)

        # Keras classification modelleri .predict() kullanır
        # Çıktı genelde (1, num_classes) şeklindedir ve zaten softmax uygulanmıştır
        prediction = self.model.predict(img_transformed)[0]  # Batch'ten kurtul

        # tf.math.top_k, PyTorch'taki torch.topk'in eşdeğeridir
        scores_tf, indices_tf = tf.math.top_k(prediction, k=int(self.obj.num_predictions))

        # TensorFlow tensörlerini NumPy dizilerine dönüştür
        scores = scores_tf.numpy()
        indices = indices_tf.numpy()

        height, width = img.shape[:2]
        bounding_box = BoundingBox(left=0, top=0, width=width, height=height)

        for score, index in zip(scores, indices):
            class_id = index.item()
            detection = Detection(
                boundingBox=bounding_box,
                # Keras modelleri genelde zaten softmax'lı (olasılık) döndürür
                # prediction[class_id] doğrudan skordur.
                confidence=prediction[class_id].item(),
                classId=class_id,
                # NOT: Keras/ImageNet etiketleri 0-indekslidir
                classLabel=self.return_label(class_id + 1),
                # TF Hub 1-indeksli, Keras 0-indeksli olabilir. Eşlemeye dikkat!
                imgUID=uID
            )
            self.obj.predictions.append(detection)


class TensorFlowSegmenter(TensorFlowBase):
    def __init__(self, obj: object):
        super().__init__(obj, tf_model_map, obj.config_model_type == CONFIG["CustomWeight"])

    def predict(self, img: np.ndarray, uID: str) -> None:
        input_tensor = self.preprocess(img)

        if self.obj.config_type == CONFIG["Semantic"]:
            # TF Hub modelleri (veya Keras) farklı imzalara sahip olabilir
            # Modelin belgelerine göre 'default' veya 'serving_default' kullanılır
            if hasattr(self.model, 'signatures'):
                prediction = self.model.signatures['default'](input_tensor)
            else:
                prediction = self.model(input_tensor, training=False)
            self._predict_semantic(prediction, img, uID)

        elif self.obj.config_type == CONFIG["Instance"]:
            prediction = self.model(input_tensor)  # TF Hub MaskRCNN
            self._predict_instance(prediction, img, uID)
        else:
            raise ValueError("config_type must be either Semantic or Instance")

    def _predict_semantic(self, prediction_dict, img, uID):
        original_h, original_w = img.shape[:2]

        # TF Hub DeepLab modeli 'logits' anahtarı altında çıktı verebilir
        # Çıktı şekli: (1, H_pred, W_pred, NumClasses)
        logits = prediction_dict['logits'][0]  # Batch'ten kurtul

        # PyTorch (N, C, H, W) kullanır, TF (N, H, W, C) kullanır. Eksen farkı!
        normalized_masks = tf.nn.softmax(logits, axis=-1)

        # Giriş tensörünün boyutları (modelin beklediği)
        h, w = logits.shape[:2]

        for class_id in range(1, normalized_masks.shape[-1]):  # Son eksen (C)
            mask_tensor = normalized_masks[..., class_id]
            mask = (mask_tensor > 0.5)

            # tf.reduce_sum ile maskenin boş olup olmadığını kontrol et
            if tf.reduce_sum(tf.cast(mask, tf.int32)) == 0:
                continue

            mask_np = mask.numpy().astype(np.uint8)
            polygons = Mask(mask_np).polygons()

            # Güven skorlarını poligon noktalarından al
            # (x, y) -> (y, x) indeksleme için
            points = polygons.points[0].astype(int)
            points_indices = points[:, [1, 0]]  # (y, x) sırası

            confidence_values = tf.gather_nd(mask_tensor, indices=points_indices)
            avg_conf = tf.reduce_mean(confidence_values).numpy().item()

            if avg_conf > self.obj.conf_threshold:
                # Keypoint'leri orijinal görüntü boyutuna ölçekle
                keypoints = [
                    KeyPoints(cx=int(x * original_w / w), cy=int(y * original_h / h), confidence=conf.item())
                    for (x, y), conf in zip(points, confidence_values.numpy())
                ]

                self.obj.predictions.append(
                    Detection(
                        boundingBox=BoundingBox(left=0, top=0, width=img.shape[1], height=img.shape[0]),
                        keyPoints=keypoints,
                        confidence=round(avg_conf, 2),
                        classId=int(class_id),
                        classLabel=self.return_label(class_id),  # COCO 1-indekslidir
                        imgUID=uID,
                        segment_type=self.obj.config_type
                    )
                )

    def _predict_instance(self, predictions, img, uID):
        boxes = predictions["detection_boxes"][0].numpy()
        labels = predictions["detection_classes"][0].numpy()
        scores = predictions["detection_scores"][0].numpy()
        masks = predictions["detection_masks"][0].numpy()
        # masks şekli: (NumDetections, MaskH, MaskW)

        original_h, original_w = img.shape[:2]

        for i in range(len(boxes)):
            if scores[i] < self.obj.conf_threshold:
                continue

            # 1. Kutuyu piksel koordinatlarına dönüştür
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1 = int(xmin * original_w), int(ymin * original_h)
            x2, y2 = int(xmax * original_w), int(ymax * original_h)
            box_w, box_h = x2 - x1, y2 - y1

            if box_w <= 0 or box_h <= 0:
                continue

            # 2. TF Hub'dan gelen küçük maskeyi (örn: 33x33) al
            small_mask = masks[i]

            # 3. Küçük maskeyi Bounding Box boyutuna yeniden boyutlandır
            # (MaskH, MaskW) -> (box_h, box_w)
            mask_resized = tf.image.resize(
                small_mask[..., tf.newaxis],  # (H, W, 1) olmalı
                (box_h, box_w)
            ).numpy().squeeze()

            # 4. Tam boyutlu bir tuval oluştur ve maskeyi doğru yere yapıştır
            full_mask = np.zeros((original_h, original_w), dtype=np.uint8)
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            full_mask[y1:y2, x1:x2] = mask_binary * 255

            # 5. Poligonları tam boyutlu maskeden çıkar
            polygons = Mask(full_mask).polygons()
            if not polygons.points:  # Poligon bulunamadıysa
                continue

            # TF Hub Mask RCNN poligon için ayrı güven skoru vermez
            keypoints = [KeyPoints(cx=int(x), cy=int(y)) for x, y in polygons.points[0]]

            class_id = int(labels[i])
            self.obj.predictions.append(
                Detection(
                    boundingBox=BoundingBox(
                        left=x1,
                        top=y1,
                        width=box_w,
                        height=box_h
                    ),
                    keyPoints=keypoints,
                    confidence=round(float(scores[i]), 2),
                    classId=class_id,
                    classLabel=self.return_label(class_id),
                    imgUID=uID,
                    segment_type=self.obj.config_type
                )
            )