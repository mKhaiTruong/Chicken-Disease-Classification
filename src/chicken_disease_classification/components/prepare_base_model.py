import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

from chicken_disease_classification.entity.entity_config import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_model(self):
        weights = models.VGG16_Weights.IMAGENET1K_V1 if self.config.params_weights == "imagenet" else None
        self.model = models.vgg16(weights=weights)
        self.save_model(self.config.base_model_path, self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, learning_rate):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False

        in_features = model.classifier[6].in_features  # 4096
        model.classifier[6] = nn.Linear(in_features, classes)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        return model, optimizer, criterion

    def update_base_model(self):
        self.full_model, self.optimizer, self.criterion = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=not self.config.params_include_top,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(self.config.updated_base_model_path, self.full_model)

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model.state_dict(), path)