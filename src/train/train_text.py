import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.preprocessing.text_preproc import clean_text, build_tfidf
from src.data.dataset import prepare_labels
from src.models.text_models import train_text_nb

def train_text_model(
    csv_path = "data/processed/movies_subset.csv"
):
    # Load data
    df = pd.read_csv(csv_path)

    # Split into train/val/test (70/15/15)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    # Prepare labels
    y = prepare_labels(df)
    y_train, y_val, y_test = (
        y[train_df.index],
        y[val_df.index],
        y[test_df.index],
    )

    # TF-IDF vectorization
    df["overview"] = df["overview"].apply(clean_text)
    x_train_tfidf, vectorizer = build_tfidf(train_df["overview"].fillna("").tolist())
    x_val_tfidf = vectorizer.transform(val_df["overview"].fillna("").tolist())
    x_test_tfidf = vectorizer.transform(test_df["overview"].fillna("").tolist())

    # Train model
    model = train_text_nb(x_train_tfidf, y_train)

    # Evaluate on validation and test
    val_preds = model.predict(x_val_tfidf)
    test_preds = model.predict(x_test_tfidf)

    print("Validation performance:")
    print(classification_report(y_val, val_preds, zero_division=0))
    print("\nTest performance:")
    print(classification_report(y_test, test_preds, zero_division=0))

    # Save outputs
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/text_train_features.npy", x_train_tfidf.toarray())
    np.save("data/processed/text_val_features.npy", x_val_tfidf.toarray())
    np.save("data/processed/text_test_features.npy", x_test_tfidf.toarray())
    np.save("data/processed/labels.npy", y)

    with open("data/processed/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("models/text_nb_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Saved TF-IDF features, labels, and trained Naive Bayes model.")


