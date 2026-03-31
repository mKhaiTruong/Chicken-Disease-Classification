import torch
import torch.nn as nn
import torch.optim as optim

from chicken_disease_classification.entity.entity_config import PrepareCallbacksConfig

class PrepareCallbacks:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config
        self.best_metric = float('inf')
        self.patience_counter = 0
        
    def save_checkpoint(self, model: nn.Module, optimizer, epoch: int, metric: float, keep_last: int = 5):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric": metric
        }
        
        path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(ckpt, path)
        
        checkpoints = sorted(self.config.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        for old_ckpt in checkpoints[:-keep_last]:
            old_ckpt.unlink()
    
    def check_early_stopping(self, metric: float) -> bool:
        if metric < self.best_metric:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.early_stopping_patience

    def save_best_model(self, model: nn.Module, metric: float) -> bool:
        if metric < self.best_metric:
            self.best_metric = metric
            torch.save(model.state_dict(), self.config.best_model_path)
            return True  # signal: best model updated
        
        return False