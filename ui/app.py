import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import joblib
import pickle
from src.models.fusion import late_fusion, predict_multilabel, predictions_to_genres
from src.preprocessing.image_preproc import extract_color_histogram, get_transforms

text_model = joblib.load("models/text_nb_model.pkl")
image_model = joblib.load("models/image_knn_model.pkl")
with open("data/processed/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

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
        image_transform = get_transforms()
        if poster_path.get():
            img = Image.open(poster_path.get()).resize((224, 224))
            img_feat = extract_color_histogram(img).reshape(1, -1)
            image_probs = image_model.predict_proba(img_feat)
        else:
            # If no image, only use text
            image_probs = np.zeros_like(text_probs)
        print(image_probs)

        fused_probs = late_fusion(text_probs, image_probs, alpha=0.6)
        preds_binary = predict_multilabel(fused_probs, threshold=0.5)
        preds_genres = predictions_to_genres(preds_binary, "data/processed/mlb.pkl")

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
    root.title("🎬 Multimodal Movie Genre Predictor")
    root.geometry("650x500")
    root.resizable(False, False)

    poster_path = tk.StringVar()
    result_text = tk.StringVar()

    # Header
    header = tk.Label(root, text="Multimodal Movie Genre Prediction", font=("Helvetica", 16, "bold"))
    header.pack(pady=10)

    # Text input frame
    text_frame = tk.Frame(root, padx=10, pady=10)
    text_frame.pack(fill="x")
    tk.Label(text_frame, text="Movie Description:", font=("Helvetica", 12)).pack(anchor="w")
    text_entry = tk.Text(text_frame, height=5, width=70, font=("Helvetica", 11))
    text_entry.pack(pady=5)

    # Poster input frame
    poster_frame = tk.Frame(root, padx=10, pady=10)
    poster_frame.pack(fill="x")
    tk.Button(poster_frame, text="Browse Poster Image", command=browse_image, bg="#4CAF50", fg="white",
              font=("Helvetica", 11)).pack(side="left")
    poster_label = tk.Label(poster_frame, bd=2, relief="sunken")
    poster_label.pack(side="left", padx=15)

    # Predict button
    tk.Button(root, text="Predict Genres", command=predict_movie, bg="#2196F3", fg="white",
              font=("Helvetica", 12, "bold")).pack(pady=15)

    # Result display
    result_display = tk.Label(root, textvariable=result_text, fg="blue", font=("Helvetica", 12, "bold"))
    result_display.pack(pady=10)

    root.mainloop()
