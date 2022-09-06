# Spotify Music Recommender
This tool generates song recommendations based on the audio features of at least one given song.

The tool uses Kaggle's 2017 Spotify dataset to cluster song data using the k-means algorithm. The songs are clustered based on audio features such as speechiness, energy, danceability, instrumentalness etc.

Once clustered, the tool uses Spotify's Web API to retrieve the target song's audio features, and uses cosine similarity to identify its closest 'neighbours': The most similar songs. 

The tool also provides some helpful data visualizations of clustered genres and clustered songs. It uses t-SNE and PCA dimensional reduction to represent the multi-datapoint audio feature vectors on a 2 dimensional coordinate plane.

# Usage
1. Update your environment variables to include SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET, which can be retrieved from developer.spotify.com
2. From your command line prompt, cd into the project directory.
3. Run the following command: python3 main.py "[SONG NAME]" "[RELEASE YEAR]"

# Further extension
I am currently in the process of converting the tool to a web-app by building a front-end for it using Flask.
