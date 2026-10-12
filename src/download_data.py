import os
import ssl
import urllib.request
import pandas as pd
import io

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATASETS = {
    "diabetes.csv": "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "heart.csv": "https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv",
    "parkinsons.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
}

def download_datasets():
    ctx = ssl._create_unverified_context()
    print("Starting dataset download...")
    for filename, url in DATASETS.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            print(f"[FOUND] {filename} already exists at {filepath}")
            continue
        print(f"[DOWNLOADING] {filename} from {url}...")
        try:
            req = urllib.request.urlopen(url, context=ctx, timeout=20)
            content = req.read()
            df = pd.read_csv(io.BytesIO(content))
            df.to_csv(filepath, index=False)
            print(f"[SUCCESS] Saved {filename} ({df.shape[0]} rows, {df.shape[1]} columns)")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            raise e

if __name__ == "__main__":
    download_datasets()
