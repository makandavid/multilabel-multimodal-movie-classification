import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from scipy.sparse import save_npz
from src.preprocessing.text_preproc import clean_text, build_tfidf
from src.data.dataset import prepare_labels
from src.models.text_models import train_text_nb

def train_text_model(csv_path = "data/processed/movies_subset.csv"):

    # Load data
    df = pd.read_csv(csv_path)
    df["overview"] = df["overview"].apply(clean_text)

    # Split into train/val/test (70/15/15)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    # Prepare labels
    y_train, mlb = prepare_labels(train_df, fit_mlb=True)
    y_val = prepare_labels(val_df, fit_mlb=False, mlb=mlb)
    y_test = prepare_labels(test_df, fit_mlb=False, mlb=mlb)

    # TF-IDF vectorization
    x_train_tfidf, vectorizer = build_tfidf(train_df["overview"].fillna("").tolist())
    x_val_tfidf = vectorizer.transform(val_df["overview"].fillna("").tolist())
    x_test_tfidf = vectorizer.transform(test_df["overview"].fillna("").tolist())

    # Train model
    model = train_text_nb(x_train_tfidf, y_train)

    # Evaluate on validation and test
    val_preds = model.predict(x_val_tfidf)
    test_preds = model.predict(x_test_tfidf)

    print("Validation metrics:")
    print(f"F1-score (micro): {f1_score(y_val, val_preds, average='micro'):.4f}")
    print(f"Hamming loss: {hamming_loss(y_val, val_preds):.4f}")
    print(f"Accuracy per label: {accuracy_score(y_val, val_preds):.4f}\n")

    print("Test metrics:")
    print(f"F1-score (micro): {f1_score(y_test, test_preds, average='micro'):.4f}")
    print(f"Hamming loss: {hamming_loss(y_test, test_preds):.4f}")
    print(f"Accuracy per label: {accuracy_score(y_test, test_preds):.4f}\n")

    # Save outputs
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    save_npz("data/processed/text_train_features.npz", x_train_tfidf)
    save_npz("data/processed/text_val_features.npz", x_val_tfidf)
    save_npz("data/processed/text_test_features.npz", x_test_tfidf)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/y_test.npy", y_test)


    with open("data/processed/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("models/text_nb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    joblib.dump(mlb, "data/processed/mlb.pkl")

    print("Saved TF-IDF features, labels, and trained Naive Bayes model.")


