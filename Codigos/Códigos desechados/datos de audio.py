import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ruta a la carpeta principal del dataset
DATASET_PATH = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD total'  # <- CAMBIA esto

# Inicializar lista para guardar datos
data = []

# Recorrer las carpetas de 1 a 29
for label in tqdm(range(1, 30), desc="Procesando audios"):
    folder_path = os.path.join(DATASET_PATH, str(label))
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                y, sr = librosa.load(file_path, sr=None)

                # Extraer características
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfccs_mean = np.mean(mfccs, axis=1)  # promedio por coeficiente

                rms = np.mean(librosa.feature.rms(y=y))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

                # Unir todo en un solo vector
                features = list(mfccs_mean) + [rms, zcr, centroid]

                # Agregar la etiqueta (cantidad de mentas)
                features.append(label)

                data.append(features)

            except Exception as e:
                print(f"Error con archivo {file_path}: {e}")

# Crear DataFrame
columns = [f"mfcc_{i+1}" for i in range(13)] + ['rms', 'zcr', 'centroid', 'label']
df = pd.DataFrame(data, columns=columns)

# Guardar a CSV
df.to_csv("features2.csv", index=False)
print("✅ Características guardadas en 'features2.csv'")
