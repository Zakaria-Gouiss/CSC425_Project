import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from trajectories import FEATURES, FEATURE_RANGES, N_SONGS, get_trajectory, TRAJECTORIES
from candidate_generator import optimized_knn, cleaned_tracks, scaled_features, starting_songs, find_similar_songs

def score_trajectory(candidates: pd.DataFrame, trajectory: dict) -> tuple[pd.DataFrame, float]:
    lows = np.array([FEATURE_RANGES[f][0] for f in FEATURES])
    spans = np.array([FEATURE_RANGES[f][1] - FEATURE_RANGES[f][0] for f in FEATURES])

    # targets: shape (N_SONGS, n_features), normalized to 0-1
    targets = np.column_stack([trajectory[f] for f in FEATURES])
    targets_norm = (targets - lows) / spans

    # songs: shape (n_candidates, n_features), normalized the same way
    songs_norm = (candidates[FEATURES].to_numpy(dtype=float) - lows) / spans

    # cost[i, j] = mean absolute error between target position i and candidate j
    cost = np.abs(targets_norm[:, None, :] - songs_norm[None, :, :]).mean(axis=2)

    rows, cols = linear_sum_assignment(cost)  # one unique candidate per position

    playlist = candidates.iloc[cols].copy()
    playlist["trajectory_position"] = rows
    playlist["trajectory_error"] = cost[rows, cols]
    playlist = playlist.sort_values("trajectory_position").reset_index(drop=True)

    score = float(np.clip(1 - playlist["trajectory_error"].mean(), 0, 1))
    return playlist, score


def evaluate_all_trajectories(candidates: pd.DataFrame) -> list[tuple[str, float, pd.DataFrame]]:
    if len(candidates) < N_SONGS:
        raise ValueError(
            f"Need at least {N_SONGS} unique candidates, got {len(candidates)}."
        )

    results = []
    for name, trajectory in TRAJECTORIES.items():
        playlist, score = score_trajectory(candidates, trajectory)
        results.append((name, score, playlist))

    # Sort highest score first
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def top_trajectories(
    trajectory_results: list[tuple[str, float, pd.DataFrame]],
    candidates: pd.DataFrame,
    n: int = 3
) -> list[tuple[str, float, pd.DataFrame]]:
    # Re-score using only the songs assigned under each top trajectory
    top = trajectory_results[:n]
    refined = []
    for name, score, playlist in top:
        trajectory = get_trajectory(name)
        rescored_playlist, rescored_score = score_trajectory(candidates, trajectory)
        refined.append((name, rescored_score, rescored_playlist))
    return refined


if __name__ == "__main__":
    # Build candidate pool from all three starting songs
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

    print(f"Candidates available: {len(candidates)}\n")

    # Score and rank all seven trajectories
    results = evaluate_all_trajectories(candidates)

    print("Trajectory rankings:")
    for name, score, _ in results:
        print(f"  {name}: {score:.3f}")

    # Top 3 trajectories with their assigned playlists
    top3 = top_trajectories(results, candidates, n=3)

    print("\nTop 3 trajectories:")
    for name, score, playlist in top3:
        print(f"\n--- {name} (score: {score:.3f}) ---")
        print(playlist[["track_name", "artists", "trajectory_position", "trajectory_error"]].to_string(index=False))