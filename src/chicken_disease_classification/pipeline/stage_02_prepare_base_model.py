from chicken_disease_classification.config.config_manager import ConfigManager
from chicken_disease_classification.components.prepare_base_model import PrepareBaseModel

class PrepareBaseModelPipeline():
    def __init__(self):
        pass
    
    def main(self):
        cfg_manager = ConfigManager()
        prepare_base_model_config = cfg_manager.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()