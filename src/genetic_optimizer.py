
from __future__ import annotations

import numpy as np
import pandas as pd

from trajectories import FEATURES, get_trajectory
from transition_scorer import (
    N_STARTING_SONGS,
    PLAYLIST_LENGTH,
    _find_column,
    _normalize_features,
    _track_identity,
    _trajectory_targets,
    score_playlist,
)


def _playlist_key(playlist: pd.DataFrame) -> tuple[str, ...]:
    """Return an ordered identity for a playlist, for duplicate detection."""
    return tuple(_track_identity(playlist).tolist())


def _trajectory_fit(playlist: pd.DataFrame, trajectory: dict) -> float:
    """Recalculate trajectory fit for the 17 evolved songs at their positions."""
    if len(playlist) != PLAYLIST_LENGTH:
        raise ValueError(
            f"Every playlist must contain {PLAYLIST_LENGTH} songs; got {len(playlist)}."
        )

    targets = _trajectory_targets(trajectory)
    targets_norm = _normalize_features(targets)
    songs_norm = _normalize_features(
        playlist.iloc[N_STARTING_SONGS:][FEATURES].to_numpy(dtype=float)
    )
    position_targets = targets_norm[N_STARTING_SONGS:, :]

    errors = np.abs(songs_norm - position_targets).mean(axis=1)
    return float(np.clip(1.0 - errors.mean(), 0.0, 1.0))


def _evaluate(
    playlist: pd.DataFrame,
    trajectory_name: str,
    trajectory_score: float,
    genre_col: str,
) -> dict:
    """Calculate a candidate playlist's fitness using the existing scorer."""
    trajectory = get_trajectory(trajectory_name)
    fit = _trajectory_fit(playlist, trajectory)
    scores = score_playlist(
        playlist=playlist,
        base_trajectory_score=float(trajectory_score),
        trajectory_fit=fit,
        genre_col=genre_col,
    )

    return {
        "trajectory": trajectory_name,
        "trajectory_score": float(trajectory_score),
        "playlist": playlist.reset_index(drop=True),
        "trajectory_fit": fit,
        **scores,
    }


def _repair_duplicates(
    playlist: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    starting_songs: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Replace repeated non-starting tracks with unused candidates when possible."""
    result = playlist.copy().reset_index(drop=True)
    start_ids = set(_track_identity(starting_songs.iloc[:N_STARTING_SONGS]))
    pool = candidate_pool.copy()
    pool["_identity"] = _track_identity(pool)
    pool = pool[~pool["_identity"].isin(start_ids)]
    pool = pool.drop_duplicates("_identity")

    used = set(start_ids)
    for position in range(N_STARTING_SONGS, len(result)):
        identity = _track_identity(result.iloc[[position]]).iloc[0]
        if identity not in used:
            used.add(identity)
            continue

        replacements = pool[~pool["_identity"].isin(used)]
        if replacements.empty:
            raise ValueError(
                "Not enough unique candidate tracks to repair duplicate songs. "
                "Increase the candidate pool before running the genetic algorithm."
            )

        replacement = replacements.iloc[int(rng.integers(len(replacements)))].drop(
            labels=["_identity"]
        )
        for column in result.columns:
            if column in replacement.index:
                result.at[position, column] = replacement[column]
        used.add(_track_identity(pd.DataFrame([replacement])).iloc[0])

    # The three starting songs must remain unchanged and in their original order.
    for position in range(N_STARTING_SONGS):
        for column in starting_songs.columns:
            if column in result.columns:
                result.at[position, column] = starting_songs.iloc[position][column]

    return result


def _crossover(
    parent_a: dict,
    parent_b: dict,
    candidate_pool: pd.DataFrame,
    starting_songs: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create a child by exchanging a segment after the fixed starting songs."""
    a = parent_a["playlist"].reset_index(drop=True)
    b = parent_b["playlist"].reset_index(drop=True)
    child = a.copy()

    # Choose a cut segment only among positions 4 through 20.
    left = int(rng.integers(N_STARTING_SONGS, PLAYLIST_LENGTH - 1))
    right = int(rng.integers(left + 1, PLAYLIST_LENGTH))
    child.iloc[left:right] = b.iloc[left:right].to_numpy()

    return _repair_duplicates(child, candidate_pool, starting_songs, rng)


def _mutate(
    playlist: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    starting_songs: pd.DataFrame,
    mutation_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Randomly replace eligible song positions with unused candidate tracks."""
    result = playlist.copy().reset_index(drop=True)
    pool = candidate_pool.copy()
    pool["_identity"] = _track_identity(pool)
    pool = pool.drop_duplicates("_identity")
    fixed_ids = set(_track_identity(starting_songs.iloc[:N_STARTING_SONGS]))
    pool = pool[~pool["_identity"].isin(fixed_ids)]

    for position in range(N_STARTING_SONGS, PLAYLIST_LENGTH):
        if rng.random() >= mutation_rate:
            continue

        used_ids = set(_track_identity(result.iloc[:position]))
        available = pool[~pool["_identity"].isin(used_ids)]
        if available.empty:
            continue

        replacement = available.iloc[int(rng.integers(len(available)))].drop(
            labels=["_identity"]
        )
        for column in result.columns:
            if column in replacement.index:
                result.at[position, column] = replacement[column]

    return _repair_duplicates(result, candidate_pool, starting_songs, rng)


def _tournament_select(
    population: list[dict],
    rng: np.random.Generator,
    tournament_size: int = 3,
) -> dict:
    """Select the highest-fitness individual from a random small tournament."""
    size = min(tournament_size, len(population))
    indices = rng.choice(len(population), size=size, replace=False)
    return max((population[int(i)] for i in indices), key=lambda item: item["score"])


def optimize_playlists(
    initial_playlists: list[dict],
    candidates: pd.DataFrame,
    starting_songs: pd.DataFrame,
    generations: int = 40,
    mutation_rate: float = 0.08,
    random_state: int = 42,
) -> dict:
    """Evolve exactly the five supplied playlists and return one winner.

    The population is never expanded: each generation contains the same number
    of individuals as ``initial_playlists``. The best individual survives via
    elitism; the remaining slots are filled with crossover/mutation offspring.
    Each child inherits the trajectory and trajectory score of its fitter parent.
    """
    if not initial_playlists:
        raise ValueError("initial_playlists cannot be empty.")
    if len(starting_songs) < N_STARTING_SONGS:
        raise ValueError("At least three starting songs are required.")
    if generations < 1:
        raise ValueError("generations must be at least 1.")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1.")

    genre_col = _find_column(candidates, ["track_genre", "genre"])
    rng = np.random.default_rng(random_state)

    pool = candidates.copy()
    pool["_identity"] = _track_identity(pool)
    pool = pool.drop_duplicates("_identity").drop(columns="_identity")

    # The population is exactly the supplied five playlists (or however many
    # initial playlists were supplied). No extra individuals are manufactured.
    population = []
    for item in initial_playlists:
        population.append(
            _evaluate(
                item["playlist"].reset_index(drop=True).copy(),
                item["trajectory"],
                float(item["trajectory_score"]),
                genre_col,
            )
        )

    population_size = len(population)
    for generation in range(generations):
        population.sort(key=lambda item: item["score"], reverse=True)
        next_population = [population[0]]  # Elitism: preserve current best.

        while len(next_population) < population_size:
            parent_a = _tournament_select(population, rng, tournament_size=3)
            parent_b = _tournament_select(population, rng, tournament_size=3)

            # Keep the fitter parent's trajectory as the child's target curve.
            trajectory_parent = max(
                (parent_a, parent_b), key=lambda item: item["score"]
            )
            child = _crossover(
                parent_a, parent_b, pool, starting_songs, rng
            )
            child = _mutate(
                child, pool, starting_songs, mutation_rate, rng
            )
            next_population.append(
                _evaluate(
                    child,
                    trajectory_parent["trajectory"],
                    trajectory_parent["trajectory_score"],
                    genre_col,
                )
            )

        population = next_population
        if (generation + 1) % 10 == 0 or generation == 0:
            best = max(population, key=lambda item: item["score"])
            print(
                f"Generation {generation + 1:>2}/{generations}: "
                f"best score={best['score']:.4f} "
                f"({best['trajectory']})"
            )

    return max(population, key=lambda item: item["score"])


if __name__ == "__main__":
    """Run the complete pipeline and evolve five playlists into one winner."""

    import time

    from candidate_generator import (
        optimized_knn,
        cleaned_tracks,
        scaled_features,
        starting_songs,
        find_similar_songs,
    )
    from trajectory_scorer import evaluate_all_trajectories
    from transition_scorer import (
        N_PLAYLISTS,
        VARIANTS_PER_TRAJECTORY,
        generate_top_playlists,
    )

    start_time = time.perf_counter()

    candidates = find_similar_songs(
        optimized_knn,
        starting_songs,
        cleaned_tracks,
        scaled_features,
        n_candidates=100,
    )
    candidates = candidates.drop_duplicates(
        subset=["track_name", "artists"]
    ).reset_index(drop=True)
    print(f"Candidates available: {len(candidates)}")

    trajectory_results = evaluate_all_trajectories(candidates)
    print("\nTop 3 trajectories:")
    for name, score, _ in trajectory_results[:3]:
        print(f"  {name}: {score:.3f}")

    initial_playlists = generate_top_playlists(
        trajectory_results=trajectory_results,
        candidates=candidates,
        starting_songs=starting_songs,
        n_playlists=N_PLAYLISTS,
        variants_per_trajectory=VARIANTS_PER_TRAJECTORY,
        random_state=42,
    )
    if len(initial_playlists) != 5:
        print(
            f"\nNote: generator returned {len(initial_playlists)} playlists, "
            "so the genetic algorithm will evolve exactly that many."
        )

    print(f"\nInitial competing playlists: {len(initial_playlists)}")
    for i, result in enumerate(initial_playlists, start=1):
        print(
            f"  {i}. {result['trajectory']} | score={result['score']:.4f} | "
            f"T={result['T']:.3f}, G={result['G']:.3f}, R={result['R']:.3f}"
        )

    print("\nEvolving the existing playlists; population size stays fixed.")
    winner = optimize_playlists(
        initial_playlists=initial_playlists,
        candidates=candidates,
        starting_songs=starting_songs,
        generations=40,
        mutation_rate=0.08,
        random_state=42,
    )

    initial_best = max(initial_playlists, key=lambda item: item["score"])
    print("\n" + "=" * 72)
    print("ONE OPTIMIZED PLAYLIST — GENETIC ALGORITHM WINNER")
    print("=" * 72)
    print(f"Best initial score: {initial_best['score']:.4f}")
    print(f"Winning score:      {winner['score']:.4f}")
    print(f"Score change:       {winner['score'] - initial_best['score']:+.4f}")
    print(f"Winning trajectory: {winner['trajectory']}")
    print(
        f"T={winner['T']:.3f} | G={winner['G']:.3f} | "
        f"R={winner['R']:.3f}"
    )
    print("\nWinning playlist:")
    print(
        winner["playlist"][
            ["track_name", "artists", "track_genre"]
        ].to_string(index=False)
    )
    print(f"\nTotal runtime: {time.perf_counter() - start_time:.2f} seconds")
