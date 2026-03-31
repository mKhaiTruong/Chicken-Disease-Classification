import sys
from chicken_disease_classification import logger
from chicken_disease_classification.exception.exception import CustomException
from chicken_disease_classification.config.config_manager import ConfigManager
from chicken_disease_classification.components.prepare_base_model import PrepareBaseModel

STAGE_NAME = "Prepare Base Model Stage"
class PrepareBaseModelPipeline():
    def __init__(self):
        pass
    
    def main(self):
        cfg_manager = ConfigManager()
        prepare_base_model_config = cfg_manager.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ == "__main__":
    STAGE_NAME = "Prepare Base Model Stage"
    try:
                logger.info(f"*****************************************")
                logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
                prepare_base_model = PrepareBaseModelPipeline()
                prepare_base_model.main()
                logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
                raise CustomException(e, sys)