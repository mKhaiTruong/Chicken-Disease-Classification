import sys
from chicken_disease_classification import logger
from chicken_disease_classification.exception.exception import CustomException
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
        
if __name__ == "__main__":
        STAGE_NAME = "Training Stage"
        try:
                logger.info(f"*****************************************")
                logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
                training = TrainingPipeline()
                training.main()
                logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
                raise CustomException(e, sys)