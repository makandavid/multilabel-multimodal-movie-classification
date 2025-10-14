from torchvision import transforms, models
import torch.nn as nn
import torch

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485 ,0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    
def extract_resnet_features(images: torch.Tensor, model: nn.Module | None = None):
    if model is None:
        base = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(base.children())[:-1])
        model.eval()
    with torch.no_grad():
        feats = torch.squeeze(model(images))
    return feats