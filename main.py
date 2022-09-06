from DataProcessing import *
from FindSongs import *
from Recommender import *
import pandas as pd
import sys
import plotly.express as px

"""
This script drives the genre and song data visualization, and also the recommendation system.
"""

# Import downloaded Spotify data
spotify_data = pd.read_csv('./data/data.csv.zip')
genre_data = pd.read_csv('./data/data_by_genres.csv')
yearly_data = pd.read_csv('./data/data_by_year.csv')
# Cluster genres
genre_data, X1, cluster_pipeline = cluster_data(genre_data, type='genre')
genre_projection = tsne_reduce_dimension(genre_data, X1)
# Visualize clusters
cluster_graph = px.scatter(genre_projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
cluster_graph.show()


# Cluster songs
spotify_data, X2, cluster_pipeline = cluster_data(spotify_data, type='song')
song_projection = pca_reduce_dimension(spotify_data, X2)
# Visualize clusters
song_cluster_graph = px.scatter(song_projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
song_cluster_graph.show()


# Command line argument display formatting
name = sys.argv[1]
year = sys.argv[2]
recommended = recommend_songs([{'name': f'{name}', 'year': int(year)}], spotify_data, cluster_pipeline)
print("Recommendations are:")
names, artists = [], []
for dict in recommended:
    names.append(dict['name'])
    artists.append(dict['artists'])
for i in range(1, len(recommended) + 1):
    print(f"Recommendation {i}: {names[i - 1]} by {artists[i - 1]}")

