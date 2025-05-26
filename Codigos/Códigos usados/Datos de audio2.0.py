import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ruta a la carpeta principal del dataset (cambia esto según tu ruta)
DATASET_PATH = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD total2'

# Parámetros para normalización
TARGET_SR = 22050  # Frecuencia de muestreo estándar
DURACION_SEGUNDOS = 4  # Duración estándar de audio (en segundos)
N_SAMPLES = TARGET_SR * DURACION_SEGUNDOS

# Lista para guardar características
data = []

# Recorrer carpetas del 1 al 29
for label in tqdm(range(1, 30), desc="Procesando audios"):
    folder_path = os.path.join(DATASET_PATH, str(label))
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Cargar audio con sr fijo
                y, sr = librosa.load(file_path, sr=TARGET_SR)

                # Ajustar la duración
                if len(y) < N_SAMPLES:
                    y = np.pad(y, (0, N_SAMPLES - len(y)))
                else:
                    y = y[:N_SAMPLES]

                # Extraer características
                mfccs = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=13)
                mfccs_mean = np.mean(mfccs, axis=1)

                rms = np.mean(librosa.feature.rms(y=y))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=TARGET_SR))

                # Vector de características + etiqueta
                features = list(mfccs_mean) + [rms, zcr, centroid, label]
                data.append(features)

            except Exception as e:
                print(f"❌ Error con {file_path}: {e}")

# Crear DataFrame
columnas = [f'mfcc_{i+1}' for i in range(13)] + ['rms', 'zcr', 'centroid', 'label']
df = pd.DataFrame(data, columns=columnas)

# Guardar a CSV
df.to_csv("features_normalizadas_TOTAL.csv", index=False)
print("✅ Características guardadas en 'features_normalizadas_TOTAL.csv'")
