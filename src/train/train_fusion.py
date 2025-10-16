import numpy as np
import joblib
import pickle
from sklearn.metrics import f1_score, hamming_loss
from scipy.sparse import load_npz
from src.models.fusion import late_fusion, predict_multilabel

def run_fusion(
    text_model_path: str,
    image_model_path: str,
    text_val_features: str,
    text_test_features: str,
    image_features_path: str,
    labels_val: str,
    labels_test: str,
    vectorizer_path: str,
    alpha: float = 0.6,
    threshold: float = 0.5
):
    # --- Load models ---
    text_model = joblib.load(text_model_path)
    image_model = joblib.load(image_model_path)

    # --- Load vectorizer ---
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # --- Load text features (raw text, not TF-IDF arrays) ---
    X_text_val = load_npz(text_val_features)
    X_text_test = load_npz(text_test_features)

    # --- Transform text using the same vectorizer ---
    # X_text_val = vectorizer.transform(X_text_val_raw)
    # X_text_test = vectorizer.transform(X_text_test_raw)

    # --- Load image features from npz ---
    img_data = np.load(image_features_path)
    X_image_val = img_data["X_val"]
    X_image_test = img_data["X_test"]

    # --- Load labels ---
    y_val = np.load(labels_val)
    y_test = np.load(labels_test)

    print("Image train:", img_data["X_train"].shape)
    print("Image val:", img_data["X_val"].shape)
    print("Image test:", img_data["X_test"].shape)

    print("Text val:", X_text_val.shape)
    print("Text test:", X_text_test.shape)

    # --- Predict probabilities ---
    text_val_probs = text_model.predict_proba(X_text_val)
    image_val_probs = image_model.predict_proba(X_image_val)
    text_test_probs = text_model.predict_proba(X_text_test)
    image_test_probs = image_model.predict_proba(X_image_test)

    # --- Fuse probabilities ---
    fused_val_probs = late_fusion(text_val_probs, image_val_probs, alpha=alpha)
    fused_test_probs = late_fusion(text_test_probs, image_test_probs, alpha=alpha)

    # --- Threshold predictions ---
    val_preds = predict_multilabel(fused_val_probs, threshold)
    test_preds = predict_multilabel(fused_test_probs, threshold)

    # --- Evaluation ---
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
