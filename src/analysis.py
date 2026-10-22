#just for me to look at the data visually, to see if the scaling worked, and to see if the data is clean
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_dataset
from preprocessing import clean_data, select_features, scale_features

def analyze_scaling(
    original: pd.DataFrame,
    scaled: pd.DataFrame
) -> None:
    # Compare feature distributions before and after scaling
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    original.boxplot(ax=axes[0], rot=45)
    axes[0].set_title("Audio Features Before Scaling")
    axes[0].set_ylabel("Original Values")

    scaled.boxplot(ax=axes[1], rot=45)
    axes[1].set_title("Audio Features After StandardScaler")
    axes[1].set_ylabel("Standardized Values")

    plt.tight_layout()
    plt.show()

    # Verify that scaling worked
    print("\nMeans after scaling:")
    print(scaled.mean().round(4))

    print("\nStandard deviations after scaling:")
    print(scaled.std().round(4))


if __name__ == "__main__":
    tracks = load_dataset()
    cleaned_tracks = clean_data(tracks)
    features = select_features(cleaned_tracks)
    scaled_features = scale_features(features)

    analyze_scaling(features, scaled_features)
