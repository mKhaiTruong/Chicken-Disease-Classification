import sys
from chicken_disease_classification import logger
from chicken_disease_classification.exception.exception import CustomException
from chicken_disease_classification.config.config_manager import ConfigManager
from chicken_disease_classification.components.evaluation import Evaluation

STAGE_NAME = "Evaluation Stage"
class EvaluationPipeline():
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigManager()
        val_config = config.get_evaluation_config()
        evaluation = Evaluation(val_config)
        evaluation.load_model()
        evaluation.evaluation()
        evaluation.save_score()
        
if __name__ == "__main__":
        STAGE_NAME = "Evaluation Stage"
        try:
                logger.info(f"*****************************************")
                logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
                training = EvaluationPipeline()
                training.main()
                logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
                raise CustomException(e, sys)