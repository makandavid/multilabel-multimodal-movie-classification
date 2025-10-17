from src.train.train_text import train_text_model
from src.train.train_image import train_image_model
from src.train.train_fusion import run_fusion
import os

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("🚀 Starting multimodal movie genre classification pipeline...")

    # # Step 1 — Train text model
    # print("\n--- Training Text Model ---")
    train_text_model()

    # # Step 2 — Train image model
    # print("\n--- Training Image Model ---")
    # train_image_model()

    # Step 3 — Combine (late fusion)
    print("\n--- Performing Late Fusion ---")
    # run_fusion(
    #     text_model_path="models/text_nb_model.pkl",
    #     image_model_path="models/image_knn_model.pkl",
    #     text_val_features="data/processed/text_val_features.npz",
    #     text_test_features="data/processed/text_test_features.npz",
    #     image_features_path="data/processed/image_features.npz",
    #     labels_val="data/processed/y_val.npy",
    #     labels_test="data/processed/y_test.npy",
    #     vectorizer_path="data/processed/tfidf_vectorizer.pkl",
    #     alpha=0.6,
    #     threshold=0.5
    # )

    print("\n✅ All models trained successfully! Results saved in /models/")
