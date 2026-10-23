#cleaning and preprocessing the data, including feature selection and scaling
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_loader import load_dataset

audio_features = [ #the features we want to keep, so I will drop any other features
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
]
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=audio_features) #dropping rows with missing values in the audio features columns
    df = df.drop_duplicates(subset=["track_id,", "track_name", "artists"]) #dropping duplicates based on track_id, track_name, and artists
    df = df.reset_index(drop=True)
    return df

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[audio_features]
    return features

def scale_features(features: pd.DataFrame) -> pd.DataFrame: #simple scaling function
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return pd.DataFrame(scaled_features, columns=features.columns)

if __name__ == "__main__": 
    tracks = load_dataset()
    cleaned_tracks = clean_data(tracks)
    features = select_features(cleaned_tracks)
    scaled_features = scale_features(features)

    print("Original rows:", len(tracks))
    print("Cleaned rows:", len(cleaned_tracks))
    print("Selected feature shape:", features.shape)
    print(scaled_features.head())
    