import os
import tensorflow as tf
from sdks.novavision.src.base.logger import LoggerManager
from capsules.Tensorflow.src.utils.utils import load_storage
from sdks.novavision.src.base.application import Application
from capsules.Tensorflow.src.configs.configs import model_map


class ModelLoader:

    def __init__(self, device: str, half: str, modelweightname: str):

        self.device = self.request.get_param("configDevice")
        self.half = self.request.get_param("configHalf")
        self.modelweightname = self.request.get_param("configModelWeights")
        self.model = None

        if self.device == 'cpu':
            tf.config.set_visible_devices([], 'GPU')
            self.device_context_name = '/cpu:0'
        elif self.device == 'gpu':
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"RuntimeError setting memory growth: {e}")
            self.device_context_name = '/gpu:0'
        else:
            raise ValueError("Invalid device. Must be 'cpu' or 'gpu'.")

        if self.half.lower() == 'true':
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
        else:
            policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)

        ModelClass = model_map.get(self.modelweightname)

        if not ModelClass:
            raise ValueError(f"Model name '{self.modelweightname}' not found in model_map.")

        with tf.device(self.device_context_name):
            self.model = ModelClass(weights='imagenet')