import pandas as pd
import os
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import csv

def extractfeature(wav):
    df = pd.read_csv('ModelT_edit.csv')
    df = df[df.filename != "sample.wav"] 
    df.to_csv('ModelT_edit.csv', index=False)

    f_name = os.path.basename(wav)
    y, sr = librosa.load(wav)
    audio_file, _ = librosa.effects.trim(y)
    chromagram = librosa.feature.chroma_stft(audio_file, sr=sr)
    root_mean_square = librosa.feature.rms(audio_file)
    spectral_centroids = librosa.feature.spectral_centroid(audio_file, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(audio_file, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(audio_file, sr=sr)
    zero_crossings_rate = librosa.feature.zero_crossing_rate(audio_file)
    y_harm, y_perc = librosa.effects.hpss(audio_file)
    tempo, _ = librosa.beat.beat_track(audio_file, sr = sr)
    mfcc1 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=1)
    mfcc2 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=2)
    mfcc3 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=3)
    mfcc4 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=4)
    mfcc5 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=5)
    mfcc6 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=6)
    mfcc7 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=7)
    mfcc8 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=8)
    mfcc9 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=9)
    mfcc10 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=10)
    mfcc11 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=11)
    mfcc12 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=12)
    mfcc13 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=13)
    mfcc14 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=14)
    mfcc15 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=15)
    mfcc16 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=16)
    mfcc17 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=17)
    mfcc18 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=18)
    mfcc19 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=19)
    mfcc20 = librosa.feature.mfcc(audio_file, sr=sr, n_mfcc=20)
    fields = [f_name,
            chromagram.mean(),
            chromagram.var(),
            root_mean_square.mean(),
            root_mean_square.var(),
            spectral_centroids.mean(),
            spectral_centroids.var(),
            spectral_bandwidth.mean(),
            spectral_bandwidth.var(),
            spectral_rolloff.mean(),
            spectral_rolloff.var(),
            zero_crossings_rate.mean(),
            zero_crossings_rate.var(),
            y_harm.mean(),y_harm.var(),
            y_perc.mean(),y_perc.var(),
            tempo,
            mfcc1.mean(),
            mfcc1.var(),
            mfcc2.mean(),
            mfcc2.var(),
            mfcc3.mean(),
            mfcc3.var(),
            mfcc4.mean(),
            mfcc4.var(),
            mfcc5.mean(),
            mfcc5.var(),
            mfcc6.mean(),
            mfcc6.var(),
            mfcc7.mean(),
            mfcc7.var(),
            mfcc8.mean(),
            mfcc8.var(),
            mfcc9.mean(),
            mfcc9.var(),
            mfcc19.mean(),
            mfcc10.var(),
            mfcc11.mean(),
            mfcc11.var(),
            mfcc12.mean(),
            mfcc12.var(),
            mfcc13.mean(),
            mfcc13.var(),
            mfcc14.mean(),
            mfcc14.var(),
            mfcc15.mean(),
            mfcc15.var(),
            mfcc16.mean(),
            mfcc16.var(),
            mfcc17.mean(),
            mfcc17.var(),
            mfcc18.mean(),
            mfcc18.var(),
            mfcc19.mean(),
            mfcc19.var(),
            mfcc20.mean(),
            mfcc20.var(),
            '0']
    with open('ModelT_edit.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def Similiarity(name):
    data = pd.read_csv('ModelT_edit.csv', index_col='filename')
    labels = data[['label']]
    data = data.drop(columns=['label'])
    data_scaled = preprocessing.scale(data)
    similarity = cosine_similarity(data_scaled)
    sim_df_labels = pd.DataFrame(similarity)
    sim_df_names = sim_df_labels.set_index(labels.index)
    sim_df_names.columns = labels.index

    series = sim_df_names[name].sort_values(ascending = False)
    series = series.drop(name)

    return series.head(5)