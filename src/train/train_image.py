import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from src.data.dataset import PosterDataset, prepare_labels
from src.models.image_models import train_image_knn
from src.preprocessing.image_preproc import extract_features_from_dataset

def train_image_model(
    csv_path: str = "data/processed/movies_subset.csv",
    model_path: str = "models/image_knn.pkl"
):
    # Load data
    df = pd.read_csv(csv_path)

    # Split into train/val/test (70/15/15)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    # Prepare labels
    y = prepare_labels(df)
    dataset = PosterDataset(train_df, y)

    # Extract color histograms
    x_train, y_train = extract_features_from_dataset(dataset)
    dataset.df = val_df
    x_val, y_val = extract_features_from_dataset(dataset)
    dataset.df = test_df
    x_test, y_test = extract_features_from_dataset(dataset)

    # Train KNN
    model = train_image_knn(x_train, y_train)

    # Evaluate
    val_preds = model.predict(x_val)
    test_preds = model.predict(x_test)

    print("Validation performance:")
    print(classification_report(y_val, val_preds, zero_division=0))
    print("\nTest performance:")
    print(classification_report(y_test, test_preds, zero_division=0))

    # Save everything
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/image_train_features.npy", x_train)
    np.save("data/processed/image_val_features.npy", x_val)
    np.save("data/processed/image_test_features.npy", x_test)

    with open("models/image_knn_model.pkl", "wb") as f:
        import pickle
        pickle.dump(model, f)

    print("Saved KNN image features and model.")