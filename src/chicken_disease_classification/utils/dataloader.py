import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from chicken_disease_classification.entity.entity_config import TrainingConfig

def _is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__    # type: ignore[name-defined]
        return shell == 'ZMQInteractiveShell'  # Jupyter
    except NameError:
        return False  # standard Python

def get_base_aug(config: TrainingConfig):
    return transforms.Compose([
        transforms.Resize(config.params_image_size[:2]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_dataloaders(config: TrainingConfig):
    base = get_base_aug(config)
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(40),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize(config.params_image_size[:2]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) if config.params_is_augmentation else base

    train_dataset = datasets.ImageFolder(root=config.training_data, transform=aug)
    val_dataset   = datasets.ImageFolder(root=config.training_data, transform=base)

    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    indices = list(range(len(train_dataset)))

    num_workers = 0 if _is_notebook() else 4
    pin_memory  = not _is_notebook()

    train_loader = DataLoader(Subset(train_dataset, indices[:train_size]),
                              batch_size=config.params_batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(Subset(val_dataset, indices[train_size:]),
                              batch_size=config.params_batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader