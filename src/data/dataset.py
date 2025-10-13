import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from download_data import get_poster

def load_movies(path="data/processed/movies_subset.csv"):
    return pd.read_csv(path)

def prepare_labels(df: pd.DataFrame):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['genres'])
    return y, mlb

class PosterDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: Compose=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = get_poster(row['id'], row['poster_path'])
        if self.transforms:
            image = self.transforms(image)
        labels = row['genres']
        return image, labels