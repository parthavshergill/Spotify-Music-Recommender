# Import packages for data analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotipy
import os

# Import packages for clustering amd visualization
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


"""
This script contains methods to cluster the audio feature datapoints and to compress them for visualization.
"""

# Set up cluster pipeline
def cluster_data(dataframe, type):
    if type == 'genre':
        cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
        # Selects all numeric columns
        X = dataframe.select_dtypes(np.number)
        # Uses these columns as vectors for k-means
        cluster_pipeline.fit(X)
        # Cluster data
        dataframe['cluster'] = cluster_pipeline.predict(X)
    elif type == 'song':
        cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20, verbose=2))],
                                         verbose=True)
        X = dataframe.select_dtypes(np.number)
        number_cols = list(X.columns)
        cluster_pipeline.fit(X)
        song_cluster_labels = cluster_pipeline.predict(X)
        dataframe['cluster_label'] = song_cluster_labels
    return dataframe, X, cluster_pipeline


# Set up PCA pipeline for visualization - we use this as each vector is in R^n where n > 2, so it is hard to visualize
def tsne_reduce_dimension(dataframe, data):
    tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=2))])
    # Compress vectors into 2 dimensional space
    genre_embedding = tsne_pipeline.fit_transform(data)
    # Store projection data in a data frame
    projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
    print(projection.head())
    projection['genres'] = dataframe['genres']
    projection['cluster'] = dataframe['cluster']
    return projection

def pca_reduce_dimension(dataframe, data):
    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
    song_embedding = pca_pipeline.fit_transform(data)
    projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
    projection['title'] = dataframe['name']
    projection['cluster'] = dataframe['cluster_label']
    return projection




