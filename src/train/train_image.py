import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from src.data.dataset import PosterDataset, prepare_labels
from src.models.image_models import save_image_model, train_image_knn
from src.preprocessing.image_preproc import extract_features_from_dataset, get_transforms

def train_image_model(csv_path: str = "data/processed/movies_valid.csv"):
    # Load data
    df = pd.read_csv(csv_path)

    # Split into train/val/test (70/15/15)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    # Prepare labels
    y_train, mlb = prepare_labels(train_df, fit_mlb=True)
    y_val = prepare_labels(val_df, fit_mlb=False, mlb=mlb)
    y_test = prepare_labels(test_df, fit_mlb=False, mlb=mlb)

    # Create datasets
    train_dataset = PosterDataset(train_df, y_train, transforms=get_transforms(train=True))
    val_dataset = PosterDataset(val_df, y_val, transforms=get_transforms(train=False))
    test_dataset = PosterDataset(test_df, y_test, transforms=get_transforms(train=False))

    # --- Extract features ---
    x_train, y_train = extract_features_from_dataset(train_dataset)
    x_val, y_val = extract_features_from_dataset(val_dataset)
    x_test, y_test = extract_features_from_dataset(test_dataset)

    # Train KNN
    model, scaler = train_image_knn(x_train, y_train)

    X_val_s = scaler.transform(x_val)
    X_test_s = scaler.transform(x_test)

    # Evaluate
    val_preds = model.predict(X_val_s)
    test_preds = model.predict(X_test_s)

    print("\nValidation metrics:")
    print(f"F1-score (micro): {f1_score(y_val, val_preds, average='micro'):.4f}")
    print(f"Hamming loss: {hamming_loss(y_val, val_preds):.4f}")
    print(f"Accuracy per label: {accuracy_score(y_val, val_preds):.4f}\n")

    print("Test metrics:")
    print(f"F1-score (micro): {f1_score(y_test, test_preds, average='micro'):.4f}")
    print(f"Hamming loss: {hamming_loss(y_test, test_preds):.4f}")
    print(f"Accuracy per label: {accuracy_score(y_test, test_preds):.4f}\n")

    # Save everything
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.savez("data/processed/image_features.npz",
             X_train=x_train, Y_train=y_train,
             X_val=x_val, Y_val=y_val,
             X_test=x_test, Y_test=y_test)

    save_image_model(model, "models/image_knn_model.pkl")
    
    print("Saved image features and trained KNN model.")
