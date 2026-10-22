# inital ML model (KNN) to find musically similar songs to initial 3
# gridsearch to find best parameters for KNN model
# plotting the results of the KNN model to visualize the neighbors of the starting songs
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from data_loader import load_dataset
from preprocessing import clean_data, select_features, scale_features
import pandas as pd

tracks = load_dataset()
cleaned_tracks = clean_data(tracks)
features = select_features(cleaned_tracks)
scaled_features = scale_features(features)

starting_songs = cleaned_tracks.sample(n=3, random_state=42) # picking 3 random valid songs

audio_features = features.columns.tolist()


def negative_mean_neighbor_distance(estimator, X, y=None):
    distances, _ = estimator.kneighbors(X)
    return -distances.mean()


baseline_knn = NearestNeighbors()
baseline_knn.fit(scaled_features)

param_grid = {
    "n_neighbors": [3, 5, 10],
    "metric": ["euclidean", "manhattan"]
}

# Use a small sample to tune the parameters
tuning_sample = scaled_features.sample(
    n=min(5000, len(scaled_features)),
    random_state=42
)

grid_search = GridSearchCV(
    estimator=NearestNeighbors(),
    param_grid=param_grid,
    scoring=negative_mean_neighbor_distance,
    cv=3,
    n_jobs=-1
)

grid_search.fit(tuning_sample)

# Fit the selected model on the full dataset
optimized_knn = NearestNeighbors(**grid_search.best_params_)
optimized_knn.fit(scaled_features)

print("Baseline KNN parameters:", baseline_knn.get_params())
print("Best KNN parameters:", grid_search.best_params_)
print("Best grid-search score:", grid_search.best_score_)


def find_similar_songs(
    model,
    starting_songs: pd.DataFrame,
    cleaned_tracks: pd.DataFrame,
    scaled_features: pd.DataFrame
) -> pd.DataFrame:
    similar_songs_indices = []
    distances_by_index = {}

    for _, song in starting_songs.iterrows():
        song_index = cleaned_tracks.index.get_loc(song.name)

        distances, indices = model.kneighbors(
            scaled_features.iloc[[song_index]],
            n_neighbors=model.n_neighbors + 1
        )

        neighbors = [
            (idx, distance)
            for idx, distance in zip(indices[0], distances[0])
            if idx != song_index
        ]

        for idx, distance in neighbors:
            if idx not in distances_by_index:
                similar_songs_indices.append(idx)
                distances_by_index[idx] = distance

    candidates = cleaned_tracks.iloc[similar_songs_indices].copy()
    candidates["knn_distance"] = [
        distances_by_index[idx] for idx in similar_songs_indices
    ]

    return candidates


def plot_nearest_neighbors(
    starting_song: pd.Series,
    cleaned_tracks: pd.DataFrame,
    baseline_neighbors: pd.DataFrame,
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
        baseline_neighbors["danceability"],
        baseline_neighbors["energy"],
        label="Baseline KNN",
        marker="o",
        s=100
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

    for _, song in baseline_neighbors.iterrows():
        plt.annotate(
            song["track_name"],
            (song["danceability"], song["energy"]),
            fontsize=7
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

    for _, song in starting_songs.iterrows():
        baseline_neighbors = find_similar_songs(
            baseline_knn,
            song.to_frame().T,
            cleaned_tracks,
            scaled_features
        )

        optimized_neighbors = find_similar_songs(
            optimized_knn,
            song.to_frame().T,
            cleaned_tracks,
            scaled_features
        )

        print(f"\nStarting song: {song['track_name']} — {song['artists']}")

        print("\nBaseline KNN neighbors:")
        print(
            baseline_neighbors[
                ["track_name", "artists", "knn_distance"] + audio_features
            ].to_string(index=False)
        )

        print("\nOptimized KNN neighbors:")
        print(
            optimized_neighbors[
                ["track_name", "artists", "knn_distance"] + audio_features
            ].to_string(index=False)
        )

        baseline_names = set(baseline_neighbors["track_name"])
        optimized_names = set(optimized_neighbors["track_name"])

        print("\nNeighbor comparison:")
        print("Shared neighbors:", baseline_names & optimized_names)
        print("Only in baseline:", baseline_names - optimized_names)
        print("Only in optimized:", optimized_names - baseline_names)

        plot_nearest_neighbors(
            song,
            cleaned_tracks,
            baseline_neighbors,
            optimized_neighbors
        )
