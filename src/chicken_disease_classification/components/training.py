import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

from chicken_disease_classification.components.prepare_callbacks import PrepareCallbacks
from chicken_disease_classification.utils.dataloader import get_dataloaders
from chicken_disease_classification.utils.engine import train_one_epoch, validate

from chicken_disease_classification.entity.entity_config import (
    TrainingConfig,
    PrepareCallbacksConfig
)

class Training:
    def __init__(self, config: TrainingConfig, callbacks_config: PrepareCallbacksConfig):
        self.config = config
        self.callbacks = PrepareCallbacks(config=callbacks_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_base_model(self):
        self.model = models.vgg16()
        self.model.classifier[6] = nn.Linear(
            self.model.classifier[6].in_features,
            self.config.params_classes
        )
        
        weights = torch.load(self.config.updated_base_model_path, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model = self.model.to(self.device)

    def get_dataloader(self):
        self.train_loader, self.val_loader = get_dataloaders(self.config)
        
    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.params_learning_rate
        )
        
        for epoch in range(self.config.params_epochs):
            train_loss = train_one_epoch(self.model, self.train_loader, criterion, optimizer, self.device)
            val_loss, accuracy = validate(self.model, self.val_loader, criterion, self.device)

            print(
                f"Epoch [{epoch+1}/{self.config.params_epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Accuracy: {accuracy:.4f}"
            )

            self.callbacks.save_checkpoint(self.model, optimizer, epoch, val_loss)
            self.callbacks.save_best_model(self.model, val_loss)

            if self.callbacks.check_early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break

        torch.save(self.model.state_dict(), self.config.trained_model_path)