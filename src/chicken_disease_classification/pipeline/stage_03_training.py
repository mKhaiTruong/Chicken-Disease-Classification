from chicken_disease_classification.config.config_manager import ConfigManager
from chicken_disease_classification.components.training import Training

STAGE_NAME = "Training Stage"
class TrainingPipeline():
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigManager()
    
        # Callbacks
        prepare_callbacks_config = config.get_prepare_callbacks_config()
        
        # Training
        training_config = config.get_training_config()
        training = Training(config=training_config, callbacks_config=prepare_callbacks_config)
        training.get_base_model()
        training.get_dataloader()
        training.train()