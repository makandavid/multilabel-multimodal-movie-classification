import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset
from src.data.download_data import get_poster

def load_movies(path="data/processed/movies_subset.csv"):
    return pd.read_csv(path)

def prepare_labels(df: pd.DataFrame):
    df["genres"] = df["genres"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['genres'] = df['genres'].apply(lambda lst: [g["name"] for g in lst if isinstance(g, dict)])
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(df['genres'])

class PosterDataset(Dataset):
    def __init__(self, df: pd.DataFrame, y: np.ndarray, transforms=None):
        self.df = df
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = get_poster(row['id'], row['poster_path'])
        if self.transforms is not None:
            image = self.transforms(image)
        label = torch.tensor(self.y[index], dtype=torch.float32)
        return image, label