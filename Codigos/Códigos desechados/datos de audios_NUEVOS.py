import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ruta al dataset
DATASET_PATH = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD total'  # Cambia esto según tu ruta

# Parámetros
TARGET_SR = 22050
DURACION_SEGUNDOS = 3
INICIO_SEGUNDOS = 0.8
FIN_SEGUNDOS = 3.0
N_SAMPLES_TOTAL = TARGET_SR * DURACION_SEGUNDOS

# Lista para características
data = []

for label in tqdm(range(1, 30), desc="Procesando audios"):
    folder_path = os.path.join(DATASET_PATH, str(label))
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                y, sr = librosa.load(file_path, sr=TARGET_SR)

                # Cortar fragmento desde 0.8s hasta 3.0s
                start_sample = int(INICIO_SEGUNDOS * sr)
                end_sample = int(FIN_SEGUNDOS * sr)
                y = y[start_sample:end_sample]

                # Rellenar si es muy corto
                if len(y) < (end_sample - start_sample):
                    y = np.pad(y, (0, (end_sample - start_sample) - len(y)))

                # MFCCs
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfccs_mean = np.mean(mfccs, axis=1)

                # Espectro de potencia
                S = np.abs(librosa.stft(y))**2
                S_db = librosa.power_to_db(S)
                spectrogram_mean = np.mean(S_db, axis=1)[:10]  # Usamos primeras 10 bandas

                # Roll-off
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

                # Bandwidth
                bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

                # Flatness
                flatness = np.mean(librosa.feature.spectral_flatness(y=y))

                # Chroma (opcional)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)[:5]  # Primeras 5

                # Energía total
                energy = np.sum(y**2)

                # Vector de características
                features = (
                    list(mfccs_mean) +
                    list(spectrogram_mean) +
                    [rolloff, bandwidth, flatness] +
                    list(chroma_mean) +
                    [energy, label]
                )
                data.append(features)

            except Exception as e:
                print(f"❌ Error con {file_path}: {e}")

# Nombres de columnas
columnas = (
    [f'mfcc_{i+1}' for i in range(13)] +
    [f'spec_mean_{i+1}' for i in range(10)] +
    ['rolloff', 'bandwidth', 'flatness'] +
    [f'chroma_{i+1}' for i in range(5)] +
    ['energy', 'label']
)

df = pd.DataFrame(data, columns=columnas)

# Guardar CSV
df.to_csv("features_desacopladas.csv", index=False)
print("✅ CSV guardado como 'features_desacopladas.csv'")
