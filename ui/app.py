import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import joblib
import pickle
import cv2
from src.models.fusion import late_fusion, predict_multilabel, predictions_to_genres
from src.preprocessing.image_preproc import extract_color_histogram

# --- Load models and vectorizer ---
text_model = joblib.load("models/text_nb_model.pkl")
image_model = joblib.load("models/image_knn_model.pkl")
with open("data/processed/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# --- Functions ---

def predict_movie():
    try:
        text_input = text_entry.get("1.0", tk.END).strip()
        if not text_input:
            messagebox.showwarning("Input needed", "Please enter movie description.")
            return

        # Text features
        X_text = vectorizer.transform([text_input])
        text_probs = text_model.predict_proba(X_text)

        # Image features
        if poster_path.get():
            img = Image.open(poster_path.get()).resize((224, 224))
            img_feat = extract_color_histogram(img).reshape(1, -1)
            image_probs = image_model.predict_proba(img_feat)
        else:
            # If no image, only use text
            image_probs = np.zeros_like(text_probs)

        # Fusion
        fused_probs = late_fusion(text_probs, image_probs, alpha=0.6)
        preds_binary = predict_multilabel(fused_probs, threshold=0.3)
        preds_genres = predictions_to_genres(preds_binary, "data/processed/mlb.pkl")


        # Display result
        result_text.set(f"Predicted genres: {preds_genres}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def browse_image():
    filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if filename:
        poster_path.set(filename)
        img = Image.open(filename).resize((150, 150))
        img_tk = ImageTk.PhotoImage(img)
        poster_label.config(image=img_tk)
        poster_label.image = img_tk

# --- GUI Setup ---
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Multimodal Movie Genre Prediction")

    poster_path = tk.StringVar()
    result_text = tk.StringVar()

    tk.Label(root, text="Movie Description:").pack(anchor="w")
    text_entry = tk.Text(root, height=5, width=60)
    text_entry.pack()

    tk.Button(root, text="Browse Poster Image", command=browse_image).pack(pady=5)
    poster_label = tk.Label(root)
    poster_label.pack()

    tk.Button(root, text="Predict Genres", command=predict_movie).pack(pady=10)
    tk.Label(root, textvariable=result_text, fg="blue").pack(pady=5)

    root.mainloop()
