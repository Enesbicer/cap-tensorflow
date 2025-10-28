import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sdks.novavision.src.base.logger import LoggerManager
from capsules.Tensorflow.src.utils.utils import load_storage
from sdks.novavision.src.base.application import Application
from capsules.Tensorflow.src.configs.configs import model_map


class ModelLoader:
    def __init__(self, config: dict):
        self.config = config
        self.application = Application()
        self.logger = LoggerManager()
        self.executor = self.application.get_param(config=config, name="ConfigExecutor")["name"]

    def _get_weight_name(self):
        """Classification weights parametresini al"""
        return self.application.get_param(config=self.config, name="classificationWeights")

    def _select_model_device(self, model, config_device):
        """
        GPU/CPU seçimi yap ve mixed precision ayarla
        TensorFlow otomatik olarak cihaz yönetimi yapar
        """
        if config_device == "GPU" and len(tf.config.list_physical_devices('GPU')) > 0:
            # GPU varsa mixed precision kullan
            if self.application.get_param(config=self.config, name="configHalf") == "True":
                # Mixed precision policy
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                self.logger.info("Mixed precision (float16) enabled for GPU")

            # GPU'yu manuel olarak seç
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # İlk GPU'yu kullan
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                    # Memory growth aktif et (tüm GPU memory'yi baştan ayırma)
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    self.logger.info(f"Using GPU: {gpus[0]}")
                except RuntimeError as e:
                    self.logger.error(f"GPU configuration error: {e}")
        else:
            # CPU kullan
            tf.config.set_visible_devices([], 'GPU')
            self.logger.info("Using CPU for inference")

        return model

    def _load_classification_model(self):
        """Classification model yükle"""
        config_device = self.application.get_param(config=self.config, name="configDevice")
        is_custom = self.application.get_param(config=self.config, name="configClassificationModelType") == "CustomWeight"
        weight_name = self._get_weight_name()

        self.logger.info(f"Selected weight name from configuration: '{weight_name}'")

        # Model fonksiyonunu configs.py'den al
        model_func = model_map[weight_name]

        if is_custom:
            # Custom model yükle
            storage_id = self.application.get_param(config=self.config, name="storageId")
            weight_path = load_storage(storage_id)
            numclass = self.application.get_param(config=self.config, name="numClass")

            try:
                # Custom trained model'i direkt yükle
                model = keras.models.load_model(weight_path)
                self.logger.info(f"Custom model loaded from: {weight_path}")

            except Exception as e:
                self.logger.error(f"Custom model loading failed: {e}")
                self.logger.info("Creating new model with custom classification head...")

                # Base model + custom head oluştur
                base_model = model_func(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg'
                )

                # Custom classification head ekle
                x = base_model.output
                predictions = keras.layers.Dense(numclass, activation='softmax', name='predictions')(x)
                model = keras.Model(inputs=base_model.input, outputs=predictions)

                # Weights yükle
                try:
                    model.load_weights(weight_path)
                    self.logger.info(f"Custom weights loaded from: {weight_path}")
                except Exception as e:
                    self.logger.warning(f"Weight loading failed: {e}")
                    self.logger.warning("Using ImageNet pre-trained weights only")

        else:
            # Pre-trained ImageNet model yükle
            try:
                model = model_func(
                    weights='imagenet',
                    include_top=True
                )
                self.logger.info(f"Pre-trained ImageNet weights loaded for '{weight_name}'")

            except Exception as e:
                self.logger.error(f"Model loading failed: {e}")
                raise

        # Device seçimi ve optimization
        model = self._select_model_device(model, config_device)

        return {
            "model": model,
            "device": config_device
        }

    def load_models(self):
        """Model yükleme ana fonksiyonu"""
        # TensorFlow model cache dizini
        os.environ['KERAS_HOME'] = "/storage/"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TF warning'leri azalt

        # Sadece classification için model yükle
        if self.executor == "Classification":
            return self._load_classification_model()
        else:
            raise ValueError(f"Unsupported executor type: {self.executor}")