import numpy as np

import pandas as pd

from scipy.optimize import linear_sum_assignment

from trajectories import (

    FEATURES,

    FEATURE_RANGES,

    get_trajectory,

)

GENRE_FAMILIES = { #useful for genre_compatibility scoring, each genre family is a set of related genres that are considered compatible with each other

    "electronic": {

        "ambient", "idm", "breakbeat", "chicago-house",

        "deep-house", "detroit-techno", "drum-and-bass",

        "club", "garage", "hardstyle", "minimal-techno",

        "electronic", "j-dance", "progressive-house",

        "trance", "edm", "electro", "techno", "dubstep",

        "house", "trip-hop", "dub",

    },

    "pop": {

        "cantopop", "k-pop", "j-idol", "j-pop", "power-pop",

        "synth-pop", "indie-pop", "pop", "mandopop",

        "pop-film", "anime",

    },

    "rock_alternative": {

        "alt-rock", "alternative", "grunge", "goth",

        "emo", "garage", "indie", "psych-rock",

        "rock", "rock-n-roll", "hard-rock", "punk-rock",

        "punk", "ska", "british",

    },

    "metal": {

        "heavy-metal", "black-metal", "death-metal",

        "grindcore", "metalcore", "metal", "hardcore",

        "industrial",

    },

    "hip_hop_rnb": {

        "hip-hop", "r-n-b", "soul", "funk", "groove",

    },

    "blues_jazz": {

        "blues", "jazz",

    },

    "country_folk": {

        "country", "bluegrass", "honky-tonk", "folk",

        "rockabilly", "singer-songwriter", "acoustic",

    },

    "latin": {

        "latin", "latino", "reggaeton", "salsa", "tango",

        "dancehall",

    },

    "brazilian": {

        "brazil", "forro", "pagode", "sertanejo", "mpb",

        "samba",

    },

    "reggae": {

        "reggae", "dub", "dancehall", "ska",

    },

    "classical": {

        "classical", "opera", "piano", "new-age",

    },

    "world_regional": {

        "afrobeat", "iranian", "malay", "spanish",

        "french", "swedish", "turkish", "indian",

        "german", "world-music",

    },

    "soundtrack_stage": {

        "disney", "show-tunes",

    },

    "mood_context": {

        "study", "sleep", "chill", "happy", "sad",

        "romance", "party", "kids", "children", "comedy",

    },

}

# Playlist configuration

PLAYLIST_LENGTH = 20

N_STARTING_SONGS = 3

N_ADDITIONAL_SONGS = PLAYLIST_LENGTH - N_STARTING_SONGS

N_PLAYLISTS = 5

# Number of randomized candidate pools to evaluate per trajectory.

VARIANTS_PER_TRAJECTORY = 100

# Weighted playlist scoring algorithm

WEIGHTS = {
    "trajectory": 0.4705882353,
    "genre": 0.2352941176,
    "transition": 0.2941176471,
}

# Audio features used to measure transition smoothness.

TRANSITION_FEATURES = [

    "tempo",

    "energy",

    "danceability",

    "valence",

]

# Fallback ranges when a feature is not in FEATURE_RANGES.

DEFAULT_RANGES = {

    "tempo": (0, 250),

    "energy": (0, 1),

    "danceability": (0, 1),

    "valence": (0, 1),

}

def _find_column(df: pd.DataFrame, names: list[str]) -> str:

    """Find a column using possible column names."""

    for name in names:

        if name in df.columns:

            return name

    raise ValueError(

        f"Could not find any of these columns: {names}"

    )

def _track_identity(df: pd.DataFrame) -> pd.Series:

    """Create a track identifier for deduplication."""

    name_col = _find_column(

        df, ["track_name", "name"]

    )

    artist_col = _find_column(

        df, ["artists", "artist"]

    )

    return (

        df[name_col].fillna("").astype(str).str.lower().str.strip()

        + " - "

        + df[artist_col].fillna("").astype(str).str.lower().str.strip()

    )

def _trajectory_targets(

    trajectory: dict,

    n_positions: int = PLAYLIST_LENGTH,

) -> np.ndarray:

    """

    Interpolate the existing trajectory targets to 20 positions.

    Returns:

        Array of shape (20, number of audio features).

    """

    targets = np.column_stack([

        np.asarray(trajectory[f], dtype=float)

        for f in FEATURES

    ])

    old_positions = np.linspace(

        0, 1, len(targets)

    )

    new_positions = np.linspace(

        0, 1, n_positions

    )

    return np.column_stack([

        np.interp(

            new_positions,

            old_positions,

            targets[:, i],

        )

        for i in range(len(FEATURES))

    ])

def _normalize_features(values: np.ndarray) -> np.ndarray:

    """Normalize audio features using the project's feature ranges."""

    lows = np.array([

        FEATURE_RANGES[f][0] for f in FEATURES

    ])

    spans = np.array([

        FEATURE_RANGES[f][1] - FEATURE_RANGES[f][0]

        for f in FEATURES

    ])

    spans = np.where(spans == 0, 1, spans)

    return np.clip(

        (values - lows) / spans,

        0,

        1,

    )

def _genre_tokens(value) -> set[str]:

    """Convert a genre value into a set of genre labels."""

    if pd.isna(value):

        return set()

    if isinstance(value, (list, tuple, set)):

        values = value

    else:

        values = str(value).replace(";", ",").split(",")

    return {

        str(item).strip().lower()

        for item in values

        if str(item).strip()

    }

def _genre_family(genre: str) -> str | None:

    """Return the broad family assigned to a genre label."""

    genre = str(genre).strip().lower()

    for family, genres in GENRE_FAMILIES.items():

        if genre in genres:

            return family

    return None

def _genre_compatibility(song_a, song_b) -> float:

    """Score compatibility between two songs' genre labels."""

    genres_a = _genre_tokens(song_a)

    genres_b = _genre_tokens(song_b)

    # Missing genre data: neutral score.

    if not genres_a or not genres_b:

        return 0.5

    pair_scores = []

    for genre_a in genres_a:

        for genre_b in genres_b:

            # Exact genre match.

            if genre_a == genre_b:

                pair_scores.append(1.0)

                continue

            family_a = _genre_family(genre_a)

            family_b = _genre_family(genre_b)

            # Same broad family, different genre labels.

            if family_a is not None and family_a == family_b:

                pair_scores.append(0.8)

            # Different known families.

            elif family_a is not None and family_b is not None:

                pair_scores.append(0.2)

            # At least one genre has not been classified.

            else:

                pair_scores.append(0.5)

    return float(np.mean(pair_scores))

def _feature_transition_score(

    song_a: pd.Series,

    song_b: pd.Series,

) -> float:

    """Measure audio-feature continuity between adjacent songs."""

    differences = []

    for feature in TRANSITION_FEATURES:

        if feature not in song_a.index or feature not in song_b.index:

            continue

        a = pd.to_numeric(

            pd.Series([song_a[feature]]),

            errors="coerce",

        ).iloc[0]

        b = pd.to_numeric(

            pd.Series([song_b[feature]]),

            errors="coerce",

        ).iloc[0]

        if pd.isna(a) or pd.isna(b):

            continue

        low, high = FEATURE_RANGES.get(

            feature, DEFAULT_RANGES.get(feature, (0, 1))

        )

        span = high - low

        difference = abs(a - b) / span if span else 0.0

        differences.append(min(difference, 1.0))

    if not differences:

        return 0.5

    # Smoothly penalize larger feature differences.

    return float(np.exp(-3 * np.mean(differences)))

def score_transition(
    song_a: pd.Series,
    song_b: pd.Series,
    genre_col: str,
) -> dict[str, float]:
    """Score genre compatibility and audio-feature continuity for adjacent songs."""
    return {
        "genre": _genre_compatibility(
            song_a[genre_col], song_b[genre_col]
        ),
        "transition": _feature_transition_score(song_a, song_b),
    }

def score_playlist(
    playlist: pd.DataFrame,
    base_trajectory_score: float,
    trajectory_fit: float,
    genre_col: str,
) -> dict[str, float]:
    """
    Calculate trajectory fit, genre compatibility, transition smoothness,
    and the final weighted playlist score.

    T combines the original trajectory score with the fit of the
    selected 17 songs to the interpolated trajectory.
    """
    genre_scores = []
    transition_scores = []

    for i in range(len(playlist) - 1):
        scores = score_transition(
            playlist.iloc[i],
            playlist.iloc[i + 1],
            genre_col,
        )

        genre_scores.append(scores["genre"])
        transition_scores.append(scores["transition"])

    G = float(np.mean(genre_scores)) if genre_scores else 0.5
    R = float(np.mean(transition_scores)) if transition_scores else 0.5

    # Include the existing trajectory score as prior information.
    T = 0.5 * base_trajectory_score + 0.5 * trajectory_fit

    final_score = (
        WEIGHTS["trajectory"] * T
        + WEIGHTS["genre"] * G
        + WEIGHTS["transition"] * R
    )

    return {
        "T": T,
        "G": G,
        "R": R,
        "score": float(final_score),
    }

def _build_playlist(

    candidates: pd.DataFrame,

    starting_songs: pd.DataFrame,

    trajectory: dict,

    base_trajectory_score: float,

    rng: np.random.Generator,

    genre_col: str,

) -> tuple[pd.DataFrame, dict[str, float]]:

    """Generate one 20-song playlist for a given trajectory."""

    # Keep the three starting songs fixed and in their original order.

    starts = starting_songs.iloc[:N_STARTING_SONGS].copy()

    # Remove starting tracks from the pool of additional songs.

    start_ids = set(_track_identity(starts))

    pool = candidates.copy()

    pool["_track_id"] = _track_identity(pool)

    pool = pool[~pool["_track_id"].isin(start_ids)]

    pool = pool.drop_duplicates("_track_id").drop(

        columns="_track_id"

    )

    if len(pool) < N_ADDITIONAL_SONGS:

        raise ValueError(

            f"Need at least {N_ADDITIONAL_SONGS} unique additional "

            f"songs after removing starting tracks; got {len(pool)}."

        )

    # Sample a subset so repeated attempts explore different candidates.

    sampled = pool.iloc[

        rng.choice(

            len(pool),

            size=N_ADDITIONAL_SONGS,

            replace=False,

        )

    ].copy()

    # Create 20 trajectory targets and use positions 4 through 20

    # for the 17 additional songs.

    targets = _trajectory_targets(trajectory)

    targets_norm = _normalize_features(targets)

    songs_norm = _normalize_features(

        sampled[FEATURES].to_numpy(dtype=float)

    )

    additional_targets = targets_norm[N_STARTING_SONGS:, :]

    # Cost of assigning each candidate to each trajectory position.

    cost = np.abs(

        additional_targets[:, None, :]

        - songs_norm[None, :, :]

    ).mean(axis=2)

    rows, cols = linear_sum_assignment(cost)

    selected = sampled.iloc[cols].copy()

    selected["trajectory_position"] = rows + N_STARTING_SONGS

    selected["trajectory_error"] = cost[rows, cols]

    selected = selected.sort_values(

        "trajectory_position"

    ).reset_index(drop=True)

    # The three starting tracks always precede the selected tracks.

    starts["trajectory_position"] = np.arange(N_STARTING_SONGS)

    starts["trajectory_error"] = np.nan

    playlist = pd.concat(

        [starts, selected],

        ignore_index=True,

    )

    # Score the additional songs against their assigned targets.

    trajectory_fit = float(

        np.clip(1 - selected["trajectory_error"].mean(), 0, 1)

    )

    scores = score_playlist(

        playlist,

        base_trajectory_score,

        trajectory_fit,

        genre_col,

    )

    return playlist, scores

def generate_top_playlists(

    trajectory_results: list[tuple[str, float, pd.DataFrame]],

    candidates: pd.DataFrame,

    starting_songs: pd.DataFrame,

    n_playlists: int = N_PLAYLISTS,

    variants_per_trajectory: int = VARIANTS_PER_TRAJECTORY,

    random_state: int = 42,

) -> list[dict]:

    """

    Generate and rank complete 20-song playlists from the top

    three trajectory results.

    Returns a list of dictionaries containing:

        trajectory, trajectory_score, playlist, T, G, R, score

    """

    if len(starting_songs) < N_STARTING_SONGS:

        raise ValueError("Exactly three starting songs are required.")

    if len(trajectory_results) < 3:

        raise ValueError("At least three trajectory results are required.")

    genre_col = _find_column(

        candidates, ["track_genre", "genre"]

    )

    rng = np.random.default_rng(random_state)

    all_playlists = []

    seen_playlists = set()

    # Only use the top three trajectories.

    for name, base_score, _ in trajectory_results[:3]:

        trajectory = get_trajectory(name)

        for _ in range(variants_per_trajectory):

            playlist, scores = _build_playlist(

                candidates=candidates,

                starting_songs=starting_songs,

                trajectory=trajectory,

                base_trajectory_score=base_score,

                rng=rng,

                genre_col=genre_col,

            )

            # Identify playlists by their ordered track sequence.

            playlist_id = tuple(_track_identity(playlist))

            if playlist_id in seen_playlists:

                continue

            seen_playlists.add(playlist_id)

            all_playlists.append({

                "trajectory": name,

                "trajectory_score": float(base_score),

                "playlist": playlist,

                **scores,

            })

    all_playlists.sort(

        key=lambda result: result["score"],

        reverse=True,

    )

    return all_playlists[:n_playlists]

if __name__ == "__main__":
    from trajectory_scorer import evaluate_all_trajectories
    from candidate_generator import (
        optimized_knn,
        cleaned_tracks,
        scaled_features,
        starting_songs,
        find_similar_songs,
    )

    # Build the candidate pool from all three starting songs.
    candidates = find_similar_songs(
        optimized_knn,
        starting_songs,
        cleaned_tracks,
        scaled_features,
        n_candidates=15,
    )

    candidates = candidates.drop_duplicates(
        subset=["track_name", "artists"]
    ).reset_index(drop=True)

    print(f"Candidates available: {len(candidates)}")

    # Rank all seven trajectories using the existing trajectory scorer.
    results = evaluate_all_trajectories(candidates)

    print("\nTop 3 trajectories:")
    for name, score, _ in results[:3]:
        print(f"  {name}: {score:.3f}")

    # Generate and rank five complete playlists.
    top_playlists = generate_top_playlists(
        trajectory_results=results,
        candidates=candidates,
        starting_songs=starting_songs,
        n_playlists=N_PLAYLISTS,
        variants_per_trajectory=VARIANTS_PER_TRAJECTORY,
        random_state=42,
    )

    print(f"\nGenerated {len(top_playlists)} playlists.")

    for i, result in enumerate(top_playlists, start=1):
        print(
            f"\n{'=' * 60}\n"
            f"Playlist {i}: {result['trajectory']}\n"
            f"Overall score: {result['score']:.3f}\n"
            f"T={result['T']:.3f} | "
            f"G={result['G']:.3f} | "
            f"R={result['R']:.3f}\n"
            f"{'=' * 60}"
        )

        print(
            result["playlist"][
                [
                    "track_name",
                    "artists",
                    "trajectory_position",
                    "track_genre",
                ]
            ].to_string(index=False)
        )

    # Inspect genre and transition compatibility for each adjacent pair.
    for i, result in enumerate(top_playlists, start=1):
        playlist = result["playlist"].reset_index(drop=True)

        print(f"\n{'=' * 70}")
        print(f"Playlist {i}: {result['trajectory']}")
        print(f"Overall: {result['score']:.3f}")
        print(f"{'=' * 70}")

        for position in range(len(playlist) - 1):
            song_a = playlist.iloc[position]
            song_b = playlist.iloc[position + 1]

            scores = score_transition(
                song_a,
                song_b,
                genre_col="track_genre",
            )

            print(
                f"{song_a['track_name']} -> {song_b['track_name']}\n"
                f"    Genre: {scores['genre']:.2f} | "
                f"Transition: {scores['transition']:.2f}"
            )
