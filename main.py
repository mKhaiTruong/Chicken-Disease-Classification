import os, sys

from chicken_disease_classification import logger
from chicken_disease_classification.exception.exception import CustomException

from chicken_disease_classification.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from chicken_disease_classification.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline


STAGE_NAME = "Data Ingestion Stage"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion = DataIngestionPipeline()
        data_ingestion.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        raise CustomException(e, sys)


STAGE_NAME = "Prepare Base Model Stage"
try:
        logger.info(f"*****************************************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = PrepareBaseModelPipeline()
        prepare_base_model.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        raise CustomException(e, sys)