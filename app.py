from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
from scipy.stats import skew,kurtosis
from pickle import load
import librosa 
from pychorus import find_and_output_chorus
import pickle
from flask import Flask, render_template, request
import requests
import spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
#auth = ngrok authtoken 2g0GrfqlzR8yK0Vx07R4cJB2KKi_7DsyjcRkVJKnWaSrubY52
from pyngrok import ngrok

#public_url = ngrok.connect(port = '5000')
#ssh_url = ngrok.connect(22,"tcp")

#public_url
#ngrok.kill()




app = Flask(__name__)

def get_song_title_from_spotify_link(link):
    # Initialize Spotipy with your credentials
    client_credentials_manager = SpotifyClientCredentials(client_id='678502a34fe24c148d0ed1de2f2a2667', client_secret='3dc0dc3c4fed4cb390eadcc151f07713')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Extract the song ID from the link
    #song_id = link.split('/')[-1]
    
    # Fetch the track information using the Spotify API
    track_info = sp.track(link)
    
    # Return the song title
    return track_info['name']


#input_file = "C:\Users\harip\OneDrive\Documents\Technocolab_Music\Choruses"

def extract_features(name):
    features = {}
    for feature in ["chroma_stft","chroma_cqt","chroma_cens","mfcc","rms","spectral_centroid","spectral_bandwidth","spectral_contrast","spectral_rolloff","tonnetz","zero_crossing_rate"]:
        if feature == "chroma_stft":
            for i in range(84):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "chroma_cqt":
            for i in range(84):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "chroma_cens":
            for i in range(84):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "mfcc":
            for i in range(140):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "rms":
            for i in range(7):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "spectral_centroid":
            for i in range(7):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "spectral_bandwidth":
            for i in range(7):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "spectral_contrast":
            for i in range(49):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "spectral_rolloff":
            for i in range(7):
                features[f"{feature}_{i+1}"] = []

        if feature == "tonnetz":
            for i in range(37):
                features[f"{feature}_{i+1}"] = []
        
        if feature == "zero_crossing_rate":
            for i in range(7):
                features[f"{feature}_{i+1}"] = []

    y, sr = librosa.load(f"Choruses/"+f"{name}")

    # Extract major audio features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

    for feature in ["chroma_stft","chroma_cqt","chroma_cens","mfcc","rms","spectral_centroid","spectral_bandwidth","spectral_contrast","spectral_rolloff","tonnetz","zero_crossing_rate"]:
        if feature == "chroma_stft":
            min_val = np.min(chroma_stft, axis=1)
            mean_val = np.mean(chroma_stft, axis=1)
            median_val = np.median(chroma_stft, axis=1)
            max_val = np.max(chroma_stft, axis=1)
            std_val = np.std(chroma_stft, axis=1)
            skew_val = skew(chroma_stft, axis=1)
            kurtosis_val = kurtosis(chroma_stft, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "chroma_cqt":
            min_val = np.min(chroma_cqt, axis=1)
            mean_val = np.mean(chroma_cqt, axis=1)
            median_val = np.median(chroma_cqt, axis=1)
            max_val = np.max(chroma_cqt, axis=1)
            std_val = np.std(chroma_cqt, axis=1)
            skew_val = skew(chroma_cqt, axis=1)
            kurtosis_val = kurtosis(chroma_cqt, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "chroma_cens":
            min_val = np.min(chroma_cens, axis=1)
            mean_val = np.mean(chroma_cens, axis=1)
            median_val = np.median(chroma_cens, axis=1)
            max_val = np.max(chroma_cens, axis=1)
            std_val = np.std(chroma_cens, axis=1)
            skew_val = skew(chroma_cens, axis=1)
            kurtosis_val = kurtosis(chroma_cens, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "mfcc":
            min_val = np.min(mfcc, axis=1)
            mean_val = np.mean(mfcc, axis=1)
            median_val = np.median(mfcc, axis=1)
            max_val = np.max(mfcc, axis=1)
            std_val = np.std(mfcc, axis=1)
            skew_val = skew(mfcc, axis=1)
            kurtosis_val = kurtosis(mfcc, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "rms":
            min_val = np.min(rms, axis=1)
            mean_val = np.mean(rms, axis=1)
            median_val = np.median(rms, axis=1)
            max_val = np.max(rms, axis=1)
            std_val = np.std(rms, axis=1)
            skew_val = skew(rms, axis=1)
            kurtosis_val = kurtosis(rms, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "spectral_centroid":
            min_val = np.min(spectral_centroid, axis=1)
            mean_val = np.mean(spectral_centroid, axis=1)
            median_val = np.median(spectral_centroid, axis=1)
            max_val = np.max(spectral_centroid, axis=1)
            std_val = np.std(spectral_centroid, axis=1)
            skew_val = skew(spectral_centroid, axis=1)
            kurtosis_val = kurtosis(spectral_centroid, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "spectral_bandwidth":
            min_val = np.min(spectral_bandwidth, axis=1)
            mean_val = np.mean(spectral_bandwidth, axis=1)
            median_val = np.median(spectral_bandwidth, axis=1)
            max_val = np.max(spectral_bandwidth, axis=1)
            std_val = np.std(spectral_bandwidth, axis=1)
            skew_val = skew(spectral_bandwidth, axis=1)
            kurtosis_val = kurtosis(spectral_bandwidth, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "spectral_contrast":
            min_val = np.min(spectral_contrast, axis=1)
            mean_val = np.mean(spectral_contrast, axis=1)
            median_val = np.median(spectral_contrast, axis=1)
            max_val = np.max(spectral_contrast, axis=1)
            std_val = np.std(spectral_contrast, axis=1)
            skew_val = skew(spectral_contrast, axis=1)
            kurtosis_val = kurtosis(spectral_contrast, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "spectral_rolloff":
            min_val = np.min(spectral_rolloff, axis=1)
            mean_val = np.mean(spectral_rolloff, axis=1)
            median_val = np.median(spectral_rolloff, axis=1)
            max_val = np.max(spectral_rolloff, axis=1)
            std_val = np.std(spectral_rolloff, axis=1)
            skew_val = skew(spectral_rolloff, axis=1)
            kurtosis_val = kurtosis(spectral_rolloff, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])

        if feature == "tonnetz":
            fmin_val = np.min(tonnetz, axis=1)
            mean_val = np.mean(tonnetz, axis=1)
            median_val = np.median(tonnetz, axis=1)
            max_val = np.max(tonnetz, axis=1)
            std_val = np.std(tonnetz, axis=1)
            skew_val = skew(tonnetz, axis=1)
            kurtosis_val = kurtosis(tonnetz, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(37):
                features[f"{feature}_{i+1}"].append(stats[i])
        
        if feature == "zero_crossing_rate":
            min_val = np.min(zero_crossing_rate, axis=1)
            mean_val = np.mean(zero_crossing_rate, axis=1)
            median_val = np.median(zero_crossing_rate, axis=1)
            max_val = np.max(zero_crossing_rate, axis=1)
            std_val = np.std(zero_crossing_rate, axis=1)
            skew_val = skew(zero_crossing_rate, axis=1)
            kurtosis_val = kurtosis(zero_crossing_rate, axis=1)
            stats = np.concatenate((min_val, mean_val, median_val, max_val, std_val, skew_val, kurtosis_val), axis=0)
            for i in range(len(stats)):
                features[f"{feature}_{i+1}"].append(stats[i])

    
    
    df = pd.DataFrame(features)
    return df




def scale_reduction(df):
    # Load scaler from pickle file
    with open("Scale.pkl", 'rb') as file:
        scaler = pickle.load(file)    
    # Scale the data
    scaled_data = scaler.transform(df)    
    # Create DataFrame with scaled data
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)    
    # Load PCA from pickle file
    with open("pca.pkl", 'rb') as file:
        pca = pickle.load(file)    
    # Perform dimensionality reduction
    data_reduced = pca.transform(df_scaled)    
    # Create DataFrame with reduced dimensionality data
    df_reduced = pd.DataFrame(data_reduced)    
    return df_reduced
    
def prediction(df):
    with open("Random_forest_model.pkl", 'rb') as file:
        model = pickle.load(file)
    
    # Make predictions using the loaded model
    pop = model.predict(df)
    
    # Convert prediction to text label
    if pop == 1:
        return "Popular"
    elif pop == 0:
        return "Not Popular"

# Homepage route
@app.route('/')
def index():
    return render_template('index.html')

# Upload song route
@app.route('/upload_songs')
def upload_song():
    return render_template('upload_songs.html')

# Spotify link route
@app.route('/spotify_link')
def spotify_link():
    return render_template('spotify_link.html')


# Prediction route for uploaded song
@app.route('/predict_song', methods=['POST'])
def predict_song():
    file = request.files['file']

    # Extract the file name
    filename = file.filename
    df = extract_features(filename)
    df = scale_reduction(df)
    predict = prediction(df)
    # Process uploaded song file and make prediction
    # Replace this with your prediction code
    return render_template('upload_songs.html', filename=filename, predict=predict)

# Prediction route for Spotify link
# Spotify link route
@app.route('/predict_spotify', methods=['POST'])
def predict_spotify():
    if request.method == 'POST':
        link = request.form.get("spotify_link")
        song_name = get_song_title_from_spotify_link(link)
        df = extract_features(f"{song_name} chorus.mp3")
        df = scale_reduction(df)
        predict = prediction(df)
        return render_template('spotify_link.html',song_name=song_name, predict=predict)
    return render_template('spotify_link.html')

if __name__ == '__main__':
    app.run(debug=True)