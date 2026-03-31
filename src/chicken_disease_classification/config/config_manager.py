from pathlib import Path

from chicken_disease_classification.constants import *
from chicken_disease_classification.utils.common import read_yaml, create_directories

from chicken_disease_classification.entity.entity_config import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    PrepareCallbacksConfig,
    TrainingConfig,
    EvaluationConfig
)

class ConfigManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion_config

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model_config
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config

    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks_config
        create_directories([config.root_dir, config.checkpoint_dir])
        
        return PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            checkpoint_dir=Path(config.checkpoint_dir),
            best_model_path=Path(config.best_model_path),
            early_stopping_patience=config.early_stopping_patience
        ) 
    
    def get_training_config(self) -> TrainingConfig:
        training_config = self.config.training_config
        create_directories([training_config.root_dir])
        
        prepare_base_model_config = self.config.prepare_base_model_config
        data_ingestion_config = self.config.data_ingestion_config
        
        training_config = TrainingConfig(
            root_dir=Path(training_config.root_dir),
            trained_model_path=Path(training_config.trained_model_path),
            updated_base_model_path=Path(prepare_base_model_config.updated_base_model_path),
            training_data=Path(data_ingestion_config.unzipped_data_dir),
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
            params_classes=self.params.CLASSES,
            params_learning_rate=self.params.LEARNING_RATE,
        )
        
        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            path_of_model=Path(self.config.prepare_callbacks_config.best_model_path),
            training_data=Path(self.config.data_ingestion_config.unzipped_data_dir),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )