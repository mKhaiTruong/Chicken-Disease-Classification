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