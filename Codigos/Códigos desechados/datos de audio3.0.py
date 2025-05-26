import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET_PATH = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD total'
TARGET_SR = 22050
DURACION_SEGUNDOS = 3
N_SAMPLES = TARGET_SR * DURACION_SEGUNDOS

data = []

for label in tqdm(range(1, 30), desc="Procesando audios"):
    folder_path = os.path.join(DATASET_PATH, str(label))
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                y, sr = librosa.load(file_path, sr=TARGET_SR)

                if len(y) < N_SAMPLES:
                    y = np.pad(y, (0, N_SAMPLES - len(y)))
                else:
                    y = y[:N_SAMPLES]

                # MFCCs
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfccs_mean = np.mean(mfccs, axis=1)

                # RMS, ZCR, Spectral
                rms = np.mean(librosa.feature.rms(y=y))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

                # Chroma
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)

                # Tonnetz
                tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
                tonnetz_mean = np.mean(tonnetz, axis=1)

                # Mel spectrogram
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
                mel_mean = np.mean(mel_spec, axis=1)[:20]  # Tomamos solo las 20 primeras bandas

                # Vector de características
                features = (
                    list(mfccs_mean) +
                    [rms, zcr, centroid, bandwidth, rolloff] +
                    list(chroma_mean) +
                    list(tonnetz_mean) +
                    list(mel_mean) +
                    [label]
                )

                data.append(features)

            except Exception as e:
                print(f"❌ Error con {file_path}: {e}")

# Nombres de columnas
columnas = (
    [f'mfcc_{i+1}' for i in range(13)] +
    ['rms', 'zcr', 'centroid', 'bandwidth', 'rolloff'] +
    [f'chroma_{i+1}' for i in range(12)] +
    [f'tonnetz_{i+1}' for i in range(6)] +
    [f'mel_{i+1}' for i in range(20)] +
    ['label']
)

df = pd.DataFrame(data, columns=columnas)
df.to_csv("features_enriquecidas.csv", index=False)
print("✅ Características enriquecidas guardadas en 'features_enriquecidas.csv'")
