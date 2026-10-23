# inital ML model (KNN) to find musically similar songs to initial 3
# gridsearch to find best parameters for KNN model
# plotting the results of the KNN model to visualize the neighbors of the starting songs
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from data_loader import load_dataset
from preprocessing import clean_data, select_features, scale_features
import pandas as pd
import os
import numpy as np
import joblib # we will use this to export our model

tracks = load_dataset()
cleaned_tracks = clean_data(tracks)
features = select_features(cleaned_tracks)
scaled_features = scale_features(features)

starting_songs = cleaned_tracks.sample(n=3, random_state=42) # picking 3 random valid songs

audio_features = features.columns.tolist()

baseline_knn = NearestNeighbors()
baseline_knn.fit(scaled_features)

genres = cleaned_tracks["track_genre"]

tuning_sample = scaled_features.sample(
    n=min(5000, len(scaled_features)),
    random_state=42
)
tuning_genres = genres.loc[tuning_sample.index].to_numpy()

def genre_match_score(estimator, X, y):
    # k+1 because the closest neighbor is the song itself, which we skip
    _, indices = estimator.kneighbors(X, n_neighbors=11)
    neighbor_genres = y[indices[:, 1:]]
    return (neighbor_genres == y[:, None]).mean()

param_grid = {
    "metric": ["euclidean", "manhattan", "cosine", "chebyshev"]
}

# One split where train = test = the whole sample, so neighbor positions line up with the genre labels
everything = np.arange(len(tuning_sample))

grid_search = GridSearchCV(
    estimator=NearestNeighbors(n_neighbors=10),
    param_grid=param_grid,
    scoring=genre_match_score,
    cv=[(everything, everything)]
)

grid_search.fit(tuning_sample, tuning_genres)

optimized_knn = NearestNeighbors(n_neighbors=10, **grid_search.best_params_)
optimized_knn.fit(scaled_features)
print("Baseline KNN parameters:", baseline_knn.get_params())
print("Best KNN parameters:", grid_search.best_params_)
print("Best grid-search score:", grid_search.best_score_)

os.makedirs("models", exist_ok=True)
joblib.dump(optimized_knn, "models/optimized_knn.pkl")  # Save the optimized KNN model to a file


def find_similar_songs(
    model,
    starting_songs: pd.DataFrame,
    cleaned_tracks: pd.DataFrame,
    scaled_features: pd.DataFrame,
    n_candidates: int = 15
) -> pd.DataFrame:

    distances_by_index = {}

    # Exclude all three starting songs from the candidate pool.
    seed_indices = {
        cleaned_tracks.index.get_loc(song_index)
        for song_index in starting_songs.index
    }

    for _, song in starting_songs.iterrows():
        song_index = cleaned_tracks.index.get_loc(song.name)

        distances, indices = model.kneighbors(
            scaled_features.iloc[[song_index]],
            n_neighbors=min(n_candidates + 1, len(cleaned_tracks))
        )

        for idx, distance in zip(indices[0], distances[0]):
            if idx in seed_indices:
                continue

            # If a song is close to multiple seeds, keep its
            # shortest distance to any starting song.
            if (
                idx not in distances_by_index
                or distance < distances_by_index[idx]
            ):
                distances_by_index[idx] = distance

    # Sort candidates from closest to farthest.
    sorted_indices = sorted(
        distances_by_index,
        key=lambda idx: distances_by_index[idx]
    )

    candidates = cleaned_tracks.iloc[sorted_indices].copy()
    candidates["knn_distance"] = [
        distances_by_index[idx] for idx in sorted_indices
    ]

    return candidates

def plot_nearest_neighbors(
    starting_song: pd.Series,
    cleaned_tracks: pd.DataFrame,
    optimized_neighbors: pd.DataFrame
):
    plt.figure(figsize=(14, 9))

    plt.scatter(
        cleaned_tracks["danceability"],
        cleaned_tracks["energy"],
        label="Full catalog",
        alpha=0.08,
        s=8
    )

    plt.scatter(
        optimized_neighbors["danceability"],
        optimized_neighbors["energy"],
        label="Optimized KNN",
        marker="^",
        s=110
    )

    plt.scatter(
        starting_song["danceability"],
        starting_song["energy"],
        label="Starting song",
        marker="*",
        s=350
    )

    for _, song in optimized_neighbors.iterrows():
        plt.annotate(
            song["track_name"],
            (song["danceability"], song["energy"]),
            fontsize=7
        )

    plt.xlabel("Danceability")
    plt.ylabel("Energy")
    plt.title(f"Nearest Neighbors of {starting_song['track_name']}")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    print("Starting songs:")
    print(starting_songs[["track_name", "artists"]])

    # Compare and plot neighbors for each starting song.
    for _, song in starting_songs.iterrows():
        optimized_neighbors = find_similar_songs(
            optimized_knn,
            song.to_frame().T,
            cleaned_tracks,
            scaled_features
        )

        print(f"\nStarting song: {song['track_name']} — {song['artists']}")

        print("\nOptimized KNN neighbors:")
        print(
            optimized_neighbors[
                ["track_name", "artists", "knn_distance"] + audio_features
            ].to_string(index=False)
        )
        plot_nearest_neighbors(
            song,
            cleaned_tracks,
            optimized_neighbors
        )

    # Build one combined candidate pool from all three starting songs.
    candidates = find_similar_songs(
        optimized_knn,
        starting_songs,
        cleaned_tracks,
        scaled_features,
        n_candidates=15
    )
    candidates = candidates.drop_duplicates(
        subset=["track_name", "artists"]
    ).reset_index(drop=True)
    top_10 = candidates.head(10)

    print(f"\nTotal unique candidates: {len(candidates)}")
    print(
        candidates[
            ["track_name", "artists", "knn_distance"] + audio_features
        ].to_string(index=False)
    )
    print("\nTop 10 candidates:")
    print(
        top_10[
            ["track_name", "artists", "knn_distance"] + audio_features
        ].to_string(index=False)
    )
    

