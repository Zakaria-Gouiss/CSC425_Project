import numpy as np
import pandas as pd


# Ten target points, one for each position in the playlist.
N_SONGS = 10

FEATURES = [
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


def line(start, end):
    """Linear progression from a starting value to an ending value."""
    return np.linspace(start, end, N_SONGS)


def steady(value):
    return np.full(N_SONGS, value, dtype=float)


def wave(low, high): #helps alternate between low and high values for a feature, creating a wave-like pattern
    return np.array([
        low, high, low + (high - low) * 0.25,
        high, low, high, low + (high - low) * 0.40,
        high, low, (low + high) / 2
    ], dtype=float)


def exploration(low, high): #ai assisted exploration of the feature space, with a focus on diversity and variation
    return np.array([
        low, low + 0.15 * (high - low),
        low + 0.40 * (high - low),
        low + 0.25 * (high - low),
        low + 0.60 * (high - low),
        low + 0.45 * (high - low),
        low + 0.80 * (high - low),
        low + 0.65 * (high - low),
        high, low + 0.75 * (high - low)
    ], dtype=float)

TRAJECTORIES = { #ai assisted possible trajectories for the playlist, each trajectory is a dictionary of features and their target values
    "steady": {
        "danceability": steady(0.60),
        "energy": steady(0.55),
        "loudness": steady(-9.0),
        "speechiness": steady(0.08),
        "acousticness": steady(0.35),
        "instrumentalness": steady(0.15),
        "liveness": steady(0.15),
        "valence": steady(0.55),
        "tempo": steady(110),
    },

    "build_up": {
        "danceability": line(0.40, 0.85),
        "energy": line(0.25, 0.90),
        "loudness": line(-18, -5),
        "speechiness": line(0.04, 0.12),
        "acousticness": line(0.65, 0.10),
        "instrumentalness": line(0.35, 0.10),
        "liveness": line(0.10, 0.30),
        "valence": line(0.35, 0.80),
        "tempo": line(80, 135),
    },

    "wind_down": {
        "danceability": line(0.85, 0.35),
        "energy": line(0.90, 0.20),
        "loudness": line(-5, -20),
        "speechiness": line(0.12, 0.04),
        "acousticness": line(0.10, 0.75),
        "instrumentalness": line(0.10, 0.45),
        "liveness": line(0.30, 0.08),
        "valence": line(0.75, 0.35),
        "tempo": line(135, 70),
    },

    "wave": {
        "danceability": wave(0.35, 0.85),
        "energy": wave(0.30, 0.90),
        "loudness": wave(-18, -5),
        "speechiness": wave(0.04, 0.16),
        "acousticness": wave(0.65, 0.10),
        "instrumentalness": wave(0.40, 0.05),
        "liveness": wave(0.08, 0.35),
        "valence": wave(0.30, 0.80),
        "tempo": wave(75, 135),
    },

    "peak_time": {
        "danceability": line(0.75, 0.92),
        "energy": line(0.80, 0.95),
        "loudness": line(-7, -4),
        "speechiness": line(0.05, 0.12),
        "acousticness": line(0.15, 0.05),
        "instrumentalness": line(0.10, 0.20),
        "liveness": line(0.15, 0.35),
        "valence": line(0.60, 0.80),
        "tempo": line(115, 135),
    },

    "emotional_arc": {
        "danceability": line(0.35, 0.65),
        "energy": np.array([
            0.35, 0.30, 0.40, 0.50, 0.60,
            0.70, 0.80, 0.75, 0.60, 0.45
        ]),
        "loudness": line(-16, -8),
        "speechiness": line(0.04, 0.08),
        "acousticness": line(0.65, 0.25),
        "instrumentalness": line(0.35, 0.10),
        "liveness": steady(0.15),
        "valence": np.array([
            0.20, 0.15, 0.25, 0.35, 0.50,
            0.65, 0.80, 0.85, 0.75, 0.90
        ]),
        "tempo": line(75, 115),
    },

    "exploration": {
        "danceability": exploration(0.35, 0.90),
        "energy": exploration(0.25, 0.90),
        "loudness": exploration(-20, -4),
        "speechiness": exploration(0.02, 0.30),
        "acousticness": exploration(0.05, 0.90),
        "instrumentalness": exploration(0.00, 0.85),
        "liveness": exploration(0.05, 0.60),
        "valence": exploration(0.10, 0.95),
        "tempo": exploration(65, 160),
    },
}

def get_trajectory(name):
    if name not in TRAJECTORIES:
        raise ValueError(
            f"Unknown trajectory '{name}'. "
            f"Choose from: {list(TRAJECTORIES.keys())}"
        )
    return TRAJECTORIES[name]


FEATURE_RANGES = {
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "loudness": (-60.0, 0.0),
    "speechiness": (0.0, 1.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "liveness": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "tempo": (0.0, 250.0),
}
