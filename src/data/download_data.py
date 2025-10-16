import os, requests, time
import pandas as pd
from tqdm import tqdm
from PIL import Image

def download_posters(df: pd.DataFrame, posters_dir="data/posters", base_url="https://image.tmdb.org/t/p/w500"):
    os.makedirs(posters_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        poster_path = row.get("poster_path")
        movie_id = row.get("id")

        if not poster_path or not isinstance(poster_path, str):
            continue

        url = f"{base_url}{poster_path}"
        out_path = f"{posters_dir}/{movie_id}.jpg"

        if os.path.exists(out_path):
            continue
        
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
            time.sleep(0.1)
        except Exception as e:
            print("failed", movie_id, e)

def get_poster(
    movie_id: int,
    poster_path: str,
    posters_dir: str = "data/posters",
    base_url: str = "https://image.tmdb.org/t/p/w500"
) -> Image.Image:
    
    os.makedirs(posters_dir, exist_ok=True)
    out_path = f"{posters_dir}/{movie_id}.jpg"

    if not poster_path or not isinstance(poster_path, str) or poster_path == "nan":
        return Image.new("RGB", (500, 750), color=(0, 0, 0))  # black placeholder
    # if not os.path.exists(out_path):
    #     url = f"{base_url}{poster_path}"
    #     try:
    #         r = requests.get(url, timeout=10)
    #         if r.status_code == 200:
    #             print(f"Poster found for movie {movie_id}")
    #             with open(out_path, "wb") as f:
    #                 f.write(r.content)
    #         else:
    #             print(f"Poster not found (HTTP {r.status_code}) for movie {movie_id}")
    #             return Image.new("RGB", (500, 750), color=(0, 0, 0))
    #     except Exception as e:
    #         print(f"Failed to download poster for movie {movie_id}: {e}")
    #         return Image.new("RGB", (500, 750), color=(0, 0, 0))

    try:
        print("Found")
        return Image.open(out_path).convert("RGB")
    except Exception as e:
        print(f"Corrupt or missing poster for movie {movie_id}: {e}")
        return Image.new("RGB", (500, 750), color=(0, 0, 0))       

def prepare_dataset(df: pd.DataFrame, num_samples=5000):
    df = df[df['poster_path'].notna()].sample(n=num_samples, random_state=42)
    download_posters(df)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/movies_subset.csv", index=False)


if __name__ == "__main__":
    df = pd.read_csv("data/raw/movies_metadata.csv", low_memory=False)
    # prepare_dataset(df, num_samples=5000)
    df_text = df[df['overview'].notna()]  # filter movies with overview
    df_text.to_csv("data/processed/movies_text.csv", index=False)