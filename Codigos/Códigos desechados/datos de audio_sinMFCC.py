import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ruta a la carpeta principal del dataset
DATASET_PATH = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD total'  # <- CAMBIA esto

# Parámetros de normalización
TARGET_SR = 22050
DURACION_SEGUNDOS = 3
N_SAMPLES = TARGET_SR * DURACION_SEGUNDOS

# Lista de características
data = []

# Recorrer carpetas del 1 al 29
for label in tqdm(range(1, 30), desc="Procesando audios"):
    folder_path = os.path.join(DATASET_PATH, str(label))
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                y, sr = librosa.load(file_path, sr=TARGET_SR)

                # Ajustar duración
                if len(y) < N_SAMPLES:
                    y = np.pad(y, (0, N_SAMPLES - len(y)))
                else:
                    y = y[:N_SAMPLES]

                # Características
                rms = np.mean(librosa.feature.rms(y=y))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=TARGET_SR))
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR))
                bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=TARGET_SR))
                flatness = np.mean(librosa.feature.spectral_flatness(y=y))

                # Chroma (promedio de los 12 coeficientes)
                chroma = librosa.feature.chroma_stft(y=y, sr=TARGET_SR)
                chroma_mean = np.mean(chroma, axis=1)

                # Spectrogram mean (solo 10 primeros bins para resumir)
                S = np.abs(librosa.stft(y))
                spec_mean = np.mean(S, axis=1)
                spec_mean = spec_mean[:10]  # 10 primeras frecuencias

                # Vector de características
                features = [rms, zcr, centroid, rolloff, bandwidth, flatness] + list(chroma_mean) + list(spec_mean) + [label]
                data.append(features)

            except Exception as e:
                print(f"❌ Error con {file_path}: {e}")

# Nombres de columnas
col_chroma = [f'chroma_{i+1}' for i in range(12)]
col_spec = [f'spec_mean_{i+1}' for i in range(10)]
columnas = ['rms', 'zcr', 'centroid', 'rolloff', 'bandwidth', 'flatness'] + col_chroma + col_spec + ['label']

# Crear y guardar DataFrame
df = pd.DataFrame(data, columns=columnas)
df.to_csv("features_sin_mfcc.csv", index=False)
print("✅ Características guardadas en 'features_sin_mfcc.csv'")
