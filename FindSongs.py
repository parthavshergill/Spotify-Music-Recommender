import spotipy
import os
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ['SPOTIFY_CLIENT_ID'],
                                                           client_secret=os.environ['SPOTIFY_CLIENT_SECRET']))


"""
This method interfaces with the Spotify Web API to retrieve tracks given the track name and the release year.
"""

def find_song(name, year):
    song_data = {}
    results = sp.search(q='track: {} year: {}'.format(name,
                                                      year), limit=1)
    if len(results['tracks']['items']) == 0:
        return None

    results = results['tracks']['items'][0]

    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)
