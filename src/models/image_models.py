import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import joblib

def train_image_knn(X_features: np.ndarray, Y: np.ndarray, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    model = OneVsRestClassifier(knn)
    model.fit(X_features, Y)
    return model

def save_image_model(model: OneVsRestClassifier, path: str):
    joblib.dump(model, path)

def load_image_model(path: str):
    return joblib.load(path)