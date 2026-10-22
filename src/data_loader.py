#basic data loading 
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"


def load_dataset() -> pd.DataFrame: #arrow function that loads our dataset from path, using pathlib to import it
    """Load the Spotify tracks dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}" 
        )

    return pd.read_csv(DATASET_PATH)


if __name__ == "__main__": #if this file is set to be called, we execute basic data loading pandas functions
    tracks = load_dataset()

    print("Dataset shape:", tracks.shape)
    print("\nColumns:")
    print(tracks.columns.tolist())

    print("\nData types:")
    print(tracks.dtypes)

    print("\nMissing values:")
    print(tracks.isnull().sum())

    print("\nSample records:")
    print(tracks.head())
