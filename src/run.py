from src.train.train_text import train_text_model
from src.train.train_image import train_image_model
from src.train.train_fusion import train_fusion_model
import os

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("🚀 Starting multimodal movie genre classification pipeline...")

    # Step 1 — Train text model
    print("\n--- Training Text Model ---")
    train_text_model()

    # Step 2 — Train image model
    print("\n--- Training Image Model ---")
    train_image_model()

    # Step 3 — Combine (late fusion)
    # print("\n--- Performing Late Fusion ---")
    # train_fusion_model()

    print("\n✅ All models trained successfully! Results saved in /models/")
