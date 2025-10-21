import os
import tensorflow as tf
from sdks.novavision.src.base.logger import LoggerManager
from capsules.Tensorflow.src.utils.utils import load_storage
from sdks.novavision.src.base.application import Application
from capsules.Tensorflow.src.configs.config import CONFIG, model_map

class ModelLoader:
    def __init__(self, config: dict):
        self.config = config
        self.application = Application()
        self.logger = LoggerManager()
        self.executor = self.application.get_param(config=config, name="ConfigExecutor")["name"]
    
    def _get_weight_name(self, default_model_key):
        if self.executor == CONFIG["Segmentation"]:
            semantic_weights = self.application.get_param(config=self.config, name="SemanticWeights")
            instance_weights = self.application.get_param(config=self.config, name="InstanceWeights")
            return semantic_weights or instance_weights
        return self.application.get_param(config=self.config, name=default_model_key)
    
    def _select_model_device(self, model, config_device):
        if config_device == "GPU" and len(tf.config.list_physical_devices('GPU')) > 0:
            if self.application.get_param(config=self.config, name="Half"):
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
        return model
    
    def _load_model_by_config(self, model_type_key: str, default_model_key: str):
        config_device = self.application.get_param(config=self.config, name="ConfigDevice")
        
        if config_device == "GPU" and len(tf.config.list_physical_devices('GPU')) > 0:
            device = '/GPU:0'
        else:
            device = '/CPU:0'
        
        is_custom = self.application.get_param(config=self.config, name=model_type_key) == CONFIG["CustomWeight"]
        weight_name = self._get_weight_name(default_model_key)
        self.logger.info(f"Selected weight name from configuration: '{weight_name}'")
        
        model_func, weights = model_map[weight_name]
        
        with tf.device(device):
            model = model_func(weights=None, include_top=False if is_custom else True)
            
            if is_custom:
                storage_id = self.application.get_param(config=self.config, name="Id")
                weight_path = load_storage(storage_id)
                try:
                    numclass = self.application.get_param(config=self.config, name="NumClass")
                    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
                    output = tf.keras.layers.Dense(numclass)(x)
                    model = tf.keras.Model(inputs=model.input, outputs=output)
                    model.load_weights(weight_path)
                except Exception as e:
                    self.logger.error(f"Weights loading failed: {e}")
            else:
                weight_path = os.path.join("/storage/", f"{weight_name}.h5")
                try:
                    model.load_weights(weight_path)
                except Exception as e:
                    self.logger.warning(f"Weights loading failed: {e}")
                    self.logger.warning("Falling back to pre-trained weights...")
                    model = model_func(weights=weights)
                    self.logger.info(f"Pre-trained '{weights}' weights loaded")
            
            model = self._select_model_device(model, config_device)
        
        return {
            "model": model,
            "device": device
        }
    
    def load_models(self):
        model_keys = {
            CONFIG["Detection"]: ("ConfigDetectionModelType", "DetectionWeights"),
            CONFIG["Classification"]: ("ConfigClassificationModelType", "ClassificationWeights"),
            CONFIG["Segmentation"]: ("ConfigSegmentationModelType", "SegmentationWeights"),
        }
        
        os.environ['TFHUB_CACHE_DIR'] = "/storage/"
        
        model_type_key, default_model_key = model_keys[self.executor]
        return self._load_model_by_config(model_type_key, default_model_key)
