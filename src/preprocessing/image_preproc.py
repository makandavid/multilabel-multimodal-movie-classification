from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm
from src.data.dataset import PosterDataset

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

def extract_color_histogram(image, bins=(8, 8, 8)):
    image = np.array(image)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_features_from_dataset(dataset: PosterDataset):
    all_features, all_labels = [], []

    for i in tqdm(range(len(dataset)), desc="Extracting color histograms"):
        image, label = dataset[i]
        hist = extract_color_histogram(image)
        all_features.append(hist)
        all_labels.append(label.numpy())

    X = np.array(all_features)
    Y = np.array(all_labels)
    return X, Y