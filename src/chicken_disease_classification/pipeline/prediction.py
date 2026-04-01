import os, sys, torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from chicken_disease_classification.config.config_manager import ConfigManager
from chicken_disease_classification.utils.dataloader import get_base_aug

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = ConfigManager()
        self.eval_cfg = cfg.get_evaluation_config()
    
    def _load_model(self):
        model = models.vgg16()
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)

        weights = torch.load(
            self.eval_cfg.path_of_model,
            map_location=self.device
        )
        model.load_state_dict(weights)
        model = model.to(self.device)
        model.eval()
        return model
    
    def predict(self):
        model = self._load_model()
        transform = get_base_aug(config=self.eval_cfg)
        
        img = Image.open(self.filename).convert("RGB")
        img = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = model(img)
            result = torch.argmax(outputs, dim=1).item()
        
        if result == 1:
            prediction = "Healthy"
        else:
            prediction = "Coccidiosis"

        return [{"image": prediction}]
        
if __name__=="__main__":
    obj = PredictionPipeline(filename=None)
    obj.predict()