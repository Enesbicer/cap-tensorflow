import os
import tensorflow as tf
from sdks.novavision.src.base.logger import LoggerManager
from capsules.Tensorflow.src.utils.utils import load_storage
from sdks.novavision.src.base.application import Application
from capsules.Tensorflow.src.configs.configs import CONFIG, model_map


class ModelLoader:
    def __init__(self, application: Application, config, logger: LoggerManager):
        self.application = application
        self.config = config
        self.logger = logger
        self.model = None
        self.device = None

    def select_model_device(self):
        """Model için device seçimi yapar"""
        try:
            config_device = self.application.get_param(
                config=self.config,
                name="configDevice"
            )

            config_half = self.application.get_param(
                config=self.config,
                name="configHalf"
            )

            # String'i boolean'a çevir
            use_mixed_precision = (config_half == "True")

            if config_device == "ConfigDeviceGPU":
                # GPU kontrolü
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    try:
                        # GPU memory growth ayarla
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)

                        self.device = "/GPU:0"
                        self.logger.info(f"GPU device seçildi: {gpus[0].name}")

                        # Mixed precision ayarı
                        if use_mixed_precision:
                            policy = tf.keras.mixed_precision.Policy('mixed_float16')
                            tf.keras.mixed_precision.set_global_policy(policy)
                            self.logger.info("Mixed precision (FP16) aktif edildi")

                    except RuntimeError as e:
                        self.logger.error(f"GPU ayarlama hatası: {str(e)}")
                        self.device = "/CPU:0"
                        self.logger.info("CPU'ya geçildi")
                else:
                    self.logger.warning("GPU bulunamadı, CPU kullanılacak")
                    self.device = "/CPU:0"

            else:  # ConfigDeviceCPU
                self.device = "/CPU:0"
                self.logger.info("CPU device seçildi")

                if use_mixed_precision:
                    self.logger.warning("Mixed precision CPU'da desteklenmiyor, göz ardı edildi")

            return self.device

        except Exception as e:
            self.logger.error(f"Device seçim hatası: {str(e)}")
            self.device = "/CPU:0"
            return self.device

    def load_model(self):
        """Model yükleme işlemini gerçekleştirir"""
        try:
            # Önce device seçimi yap
            self.select_model_device()

            # Config parametrelerini al
            model_type = self.application.get_param(
                config=self.config,
                name="ConfigClassificationModelType"
            )

            model_name = self.application.get_param(
                config=self.config,
                name="classificationWeights"
            )

            is_custom = (model_type == "ClassificationCustomWeight")

            # Seçilen device ile model yükle
            with tf.device(self.device):
                if is_custom:
                    # Custom model için
                    num_classes = self.application.get_param(
                        config=self.config,
                        name="numClass"
                    )

                    storage_id = self.application.get_param(
                        config=self.config,
                        name="storageSource"
                    )

                    weight_path = load_storage(storage_id)

                    self.logger.info(f"Custom model yükleniyor: {model_name}, Classes: {num_classes}")

                    # TensorFlow custom model yükleme
                    if model_name in model_map:
                        base_model = model_map[model_name](
                            weights=None,
                            include_top=False,
                            input_shape=(224, 224, 3)
                        )

                        # Custom classifier ekleme
                        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
                        x = tf.keras.layers.Dense(256, activation='relu')(x)
                        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

                        self.model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

                        # Ağırlıkları yükle
                        self.model.load_weights(weight_path)
                        self.logger.info(f"Custom weights yüklendi: {weight_path}")
                    else:
                        # Direkt kaydedilmiş model yükle
                        self.model = tf.keras.models.load_model(weight_path)
                        self.logger.info(f"Saved model yüklendi: {weight_path}")

                else:
                    # Pretrained model için
                    self.logger.info(f"Pretrained model yükleniyor: {model_name}")

                    if model_name in model_map:
                        self.model = model_map[model_name](
                            weights='imagenet',
                            include_top=True
                        )
                        self.logger.info(f"Pretrained model başarıyla yüklendi: {model_name}")
                    else:
                        raise ValueError(f"Model bulunamadı: {model_name}")

            self.logger.info(f"Model yükleme işlemi tamamlandı - Device: {self.device}")
            return self.model

        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {str(e)}")
            raise


def load_classification_model(application: Application, config, logger: LoggerManager):
    """Model yükleme fonksiyonu"""
    loader = ModelLoader(application, config, logger)
    return loader.load_model()