from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix
import numpy as np
import joblib

def train_text_nb(X: csr_matrix, Y: np.ndarray):
    model = OneVsRestClassifier(MultinomialNB())
    model.fit(X, Y)
    return model

def save_text_model(model: OneVsRestClassifier, path: str):
    joblib.dump(model, path)

def load_text_model(path: str):
    return joblib.load(path)