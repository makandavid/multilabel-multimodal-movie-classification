import numpy as np
import joblib
from sklearn.metrics import f1_score
from src.models.fusion import late_fusion

def train_fusion_model(
    text_model_path="models/text_nb.pkl",
    image_model_path="models/image_knn.pkl",
    mlb_path="models/label_binarizer.pkl"
):
    pass
