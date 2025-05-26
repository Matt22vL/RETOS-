import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ruta a la carpeta principal del dataset
DATASET_PATH = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD total'  # <- CAMBIA esto

# Parámetros para normalización y recorte
TARGET_SR = 22050
TIEMPO_INICIO = 0.8  # segundos
TIEMPO_FIN = 3.0     # segundos
DURACION_SEGUNDOS = TIEMPO_FIN - TIEMPO_INICIO

# Lista para guardar características
data = []

for label in tqdm(range(1, 30), desc="Procesando audios"):
    folder_path = os.path.join(DATASET_PATH, str(label))
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Cargar audio
                y, sr = librosa.load(file_path, sr=TARGET_SR)

                # Recortar entre 0.8 y 3.0 segundos
                inicio = int(TIEMPO_INICIO * sr)
                fin = int(TIEMPO_FIN * sr)
                y = y[inicio:fin]

                # Si está corto, rellenar
                if len(y) < fin - inicio:
                    y = np.pad(y, (0, (fin - inicio) - len(y)))

                # Características acústicas
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfccs_mean = np.mean(mfccs, axis=1)

                rms = np.mean(librosa.feature.rms(y=y))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

                # Detección de onsets (impactos)
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=False, units='frames')
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)

                num_onsets = len(onset_times)
                mean_interval = np.mean(np.diff(onset_times)) if len(onset_times) > 1 else 0

                # Vector de características + etiqueta
                features = list(mfccs_mean) + [rms, zcr, centroid, num_onsets, mean_interval, label]
                data.append(features)

            except Exception as e:
                print(f"❌ Error con {file_path}: {e}")

# Crear DataFrame
columnas = [f'mfcc_{i+1}' for i in range(13)] + ['rms', 'zcr', 'centroid', 'num_onsets', 'mean_interval', 'label']
df = pd.DataFrame(data, columns=columnas)

# Guardar CSV
df.to_csv("features_con_onsets.csv", index=False)
print("✅ Características guardadas en 'features_con_onsets.csv'")
