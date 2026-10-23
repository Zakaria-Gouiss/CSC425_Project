# The AI DJ: Context-Aware Music Recommendation and Playlist Optimization

## Team Members: 
- Zak Gouiss
  
## Overview

AI DJ is a music recommendation system that predicts what song should play next based on a listener's recent listening history.

Rather than simply finding songs similar to the current track, the system will analyze the **last three songs** and their musical progression to recommend a song that fits the overall sequence.

## AI Approach

The system will use:

* **KNN** to generate musically similar candidate songs
* **Trajectory analysis** to identify changes in energy, tempo, valence, and other features
* **Transition scoring** to rank candidates based on how well they fit the current sequence
* **Genetic algorithms** as a potential advanced approach for optimizing complete playlists

## Dataset

The project uses a Spotify tracks dataset containing approximately **110,000+ songs**.

Important features include:

* Energy
* Danceability
* Valence
* Tempo
* Loudness
* Acousticness
* Instrumentalness
* Key/Mode
* Genre

Preprocessing and feature scaling will be performed before recommendation.

## Input → Output

**Input:** The last three songs played by a listener.

**AI Task:** Context-aware music recommendation.

**Output:** A music playlist (starting with first 3 songs) that generate great recommended next songs to listen, with transitions.

## Technologies (make sure that these are installed to execute the program!)

* Python 3.10 + (Having a version 3.10 or higher is a MUST!)
* Jupyter Notebook
* pandas
* NumPy
* scikit-learn
* Matplotlib

## Project Status

**Milestone 1 — Proposal and Planning**

Currently preparing the dataset, project structure, and baseline recommendation approach.
