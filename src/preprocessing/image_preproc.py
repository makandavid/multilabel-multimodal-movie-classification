from torchvision import transforms, models
import torch.nn as nn
import torch
import numpy as np
import cv2

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

def extract_color_historgram(image, bins=(8, 8, 8)):
    image = np.array(image)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist