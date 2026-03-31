import sys
from chicken_disease_classification import logger
from chicken_disease_classification.exception.exception import CustomException
from chicken_disease_classification.config.config_manager import ConfigManager
from chicken_disease_classification.components.data_ingestion import DataIngestion

class DataIngestionPipeline():
    def __init__(self):
        pass
    
    def main(self):
        cfg_manager = ConfigManager()
        data_ingestion_config = cfg_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == "__main__":
        STAGE_NAME = "Data Ingestion Stage"
        try:
                logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
                data_ingestion = DataIngestionPipeline()
                data_ingestion.main()
                logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
                raise CustomException(e, sys)