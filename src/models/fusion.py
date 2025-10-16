import numpy as np

def late_fusion(text_probs: np.ndarray, image_probs: np.ndarray, alpha=0.6):
    # assert text_probs.shape == image_probs.shape
    fused = alpha * text_probs + (1 - alpha) * image_probs
    return fused

def predict_multilabel(fused_probs: np.ndarray, threshold=0.5):
    return (fused_probs >= threshold).astype(int)

