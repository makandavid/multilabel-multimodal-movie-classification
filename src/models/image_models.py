import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import joblib

def train_image_knn(X_features: np.ndarray, Y: np.ndarray, n_neighbors=5, metric='manhattan'):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric=metric)
    model = OneVsRestClassifier(knn)
    model.fit(X_scaled, Y)
    return model, scaler

def save_image_model(model: OneVsRestClassifier, path: str):
    joblib.dump(model, path)

def load_image_model(path: str):
    return joblib.load(path)