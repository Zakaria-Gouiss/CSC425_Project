import numpy as np
import pandas as pd


# Seventeen target points, one for each position in the playlist (total output is a 20 song playlist, with the first 3 songs being the starting songs, and the last 17 songs being the generated candidates)
N_SONGS = 17

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
    return np.linspace(start, end, N_SONGS, dtype=float)


def steady(value):
    return np.full(N_SONGS, value, dtype=float)


def wave(low, high):
    t = np.linspace(0, 1, N_SONGS)
    return low + (high - low) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)) #ai assisted wave function to create a sinusoidal pattern for the trajectory


def exploration(low, high): #ai assisted exploration of the feature space, with a focus on diversity and variation
    t = np.linspace(0, 1, N_SONGS)
    # Zigzag: climbs with a dip every few songs, so no two neighbors are alike
    zigzag = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) * t + 0.5 * t * 0.6
    zigzag = np.clip(zigzag, 0, 1)
    return low + (high - low) * zigzag


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
        "energy": 0.3 + 0.5 * np.sin(np.pi * np.linspace(0, 1, N_SONGS)),  # rises to a peak, then falls
        "loudness": line(-16, -8),
        "speechiness": line(0.04, 0.08),
        "acousticness": line(0.65, 0.25),
        "instrumentalness": line(0.35, 0.10),
        "liveness": steady(0.15),
        "valence": 0.2 + 0.7 * np.linspace(0, 1, N_SONGS),
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
