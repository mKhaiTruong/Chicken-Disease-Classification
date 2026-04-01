import torch
import torch.nn as nn
from torchvision import models, datasets
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from chicken_disease_classification.utils.common import save_json
from chicken_disease_classification.utils.dataloader import get_base_aug

from chicken_disease_classification.entity.entity_config import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._val_loader()
    
    def _val_loader(self):
        transform = get_base_aug(config=self.config)
        full_dataset = datasets.ImageFolder(root=self.config.training_data, transform=transform)
        val_size = int(0.2 * len(full_dataset))
        indices = list(range(len(full_dataset)))

        self.val_loader = DataLoader(
            Subset(full_dataset, indices[-val_size:]),
            batch_size=self.config.params_batch_size,
            shuffle=False
        )
        self.num_classes = len(full_dataset.classes)

    def load_model(self):
        self.model = models.vgg16()
        self.model.classifier[6] = nn.Linear(
            self.model.classifier[6].in_features,
            self.num_classes
        )
        self.model.load_state_dict(
            torch.load(self.config.path_of_model, map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluation(self):
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            total_loss += criterion(outputs, labels).item()
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)

        self.score = {
            "loss": total_loss / len(self.val_loader),
            "accuracy": correct / total
        }

    def save_score(self):
        save_json(path=Path("scores.json"), content=self.score)
        
    def log_into_mlflow(self):
        mlflow.log_metrics(self.score)
        
        # Log model + register
        mlflow.pytorch.log_model(
            self.model,
            artifact_path="model",
            registered_model_name="Chicken_Disease_Classification"
        )