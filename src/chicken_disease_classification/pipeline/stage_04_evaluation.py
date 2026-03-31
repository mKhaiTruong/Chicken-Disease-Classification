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