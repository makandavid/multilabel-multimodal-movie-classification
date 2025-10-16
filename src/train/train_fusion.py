import pickle
import numpy as np
import joblib
from sklearn.metrics import f1_score, hamming_loss
from src.models.fusion import late_fusion, predict_multilabel

def run_fusion(
    text_model_path: str,
    image_model_path: str,
    text_val_features: str,
    image_val_features: str,
    labels_val: str,
    text_test_features: str,
    image_test_features: str,
    labels_test: str,
    alpha: float = 0.6,
    threshold: float = 0.5
):
    # Load models
    text_model = joblib.load(text_model_path)
    image_model = joblib.load(image_model_path)

    with open("data/processed/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Load features
    X_text_val = np.load(text_val_features, allow_pickle=True)
    X_text_val = vectorizer.transform(X_text_val)
    X_image_val = np.load(image_val_features)
    y_val = np.load(labels_val)

    X_text_test = np.load(text_test_features)
    X_text_test = vectorizer.transform(X_text_test)
    X_image_test = np.load(image_test_features)
    y_test = np.load(labels_test)

    # Predict probabilities
    text_val_probs = text_model.predict_proba(X_text_val)
    image_val_probs = image_model.predict_proba(X_image_val)
    text_test_probs = text_model.predict_proba(X_text_test)
    image_test_probs = image_model.predict_proba(X_image_test)

    # Fuse
    fused_val_probs = late_fusion(text_val_probs, image_val_probs, alpha=alpha)
    fused_test_probs = late_fusion(text_test_probs, image_test_probs, alpha=alpha)

    # Threshold
    val_preds = predict_multilabel(fused_val_probs, threshold)
    test_preds = predict_multilabel(fused_test_probs, threshold)

    # Evaluate
    def evaluate(y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        hamming = hamming_loss(y_true, y_pred)
        acc_per_label = (y_true == y_pred).mean(axis=0).mean()
        print(f"F1-score (micro): {f1:.4f}")
        print(f"Hamming loss: {hamming:.4f}")
        print(f"Accuracy per label: {acc_per_label:.4f}")
        return f1, hamming, acc_per_label

    print("--- Validation metrics ---")
    evaluate(y_val, val_preds)
    print("\n--- Test metrics ---")
    evaluate(y_test, test_preds)
