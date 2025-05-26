import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from scipy.signal import find_peaks

# === Parámetros ===22050 
DATASET_PATH = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Codigos\\1905\\nuevos'  # Cambia esto según tu ruta
TARGET_SR = 22050
FRAME_MS = 50 # tamaño de frame para recorte por energía (en milisegundos)
ENERGY_THRESHOLD = 0.003  # umbral de energía para considerar un golpe
FRAME_SIZE = int(0.2 * TARGET_SR)  # 200 ms para extracción de características
HOP_SIZE = FRAME_SIZE  # sin superposición
N_MFCC = 13

# === Función para recortar partes sin golpes (actividad baja) ===
def recortar_a_golpes(y, sr, frame_ms, energy_threshold):
    frame_length = int(frame_ms / 1000 * sr)
    hop_length = frame_length

    energies = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T

    active_frames = []
    for i, energy in enumerate(energies):
        if energy > energy_threshold:
            start = i * hop_length
            end = start + frame_length
            active_frames.append(y[start:end])

    if len(active_frames) == 0:
        return y  # si no detectó nada, devuelve el original
    return np.concatenate(active_frames)

# === Función para extraer características por ventanas ===
def extraer_caracteristicas_ventaneadas(y, sr):
    frames = librosa.util.frame(y, frame_length=FRAME_SIZE, hop_length=HOP_SIZE).T
    caracteristicas = []

    for frame in frames:
        mfccs = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs, axis=1)

        rms = librosa.feature.rms(y=frame)[0]
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)

        zcr = librosa.feature.zero_crossing_rate(y=frame)[0]
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)

        centroid = librosa.feature.spectral_centroid(y=frame, sr=sr)[0]
        centroid_mean = np.mean(centroid)
        centroid_var = np.var(centroid)

        rolloff = librosa.feature.spectral_rolloff(y=frame, sr=sr)[0]
        rolloff_mean = np.mean(rolloff)
        rolloff_var = np.var(rolloff)

        S = np.abs(librosa.stft(frame))**2
        psd = np.mean(S, axis=1)
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else np.zeros_like(psd)
        entropia = entropy(psd_norm)

        env = librosa.onset.onset_strength(y=frame, sr=sr)
        peaks, _ = find_peaks(env, height=np.max(env)*0.3)
        num_peaks = len(peaks)

        caracteristicas.append(
            list(mfccs_mean) +
            [rms_mean, rms_var, zcr_mean, zcr_var, centroid_mean, centroid_var,
             rolloff_mean, rolloff_var, entropia, num_peaks]
        )

    caracteristicas = np.array(caracteristicas)
    media = np.mean(caracteristicas, axis=0)
    varianza = np.var(caracteristicas, axis=0)
    return np.concatenate([media, varianza])

# === Proceso principal ===
data = []
for label in tqdm(range(1, 30), desc="Procesando audios"):
    folder_path = os.path.join(DATASET_PATH, str(label))
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                y, sr = librosa.load(file_path, sr=TARGET_SR)

                # Recortar partes sin golpes antes de extraer características
                y = recortar_a_golpes(y, sr, frame_ms=FRAME_MS, energy_threshold=ENERGY_THRESHOLD)

                # Duración final ajustada
                if len(y) < FRAME_SIZE * 2:
                    continue  # muy corto tras recorte, descartar

                features = extraer_caracteristicas_ventaneadas(y, sr)
                data.append(np.concatenate([features, [label]]))

            except Exception as e:
                print(f"❌ Error con {file_path}: {e}")

# === Guardar características ===
n_features = 13 + 10  # 13 MFCC + 10 adicionales
column_names = (
    [f'mfcc_{i+1}_mean' for i in range(13)] +
    ['rms_mean', 'rms_var', 'zcr_mean', 'zcr_var', 'centroid_mean', 'centroid_var',
     'rolloff_mean', 'rolloff_var', 'entropy', 'num_peaks'] +
    [f'mfcc_{i+1}_var' for i in range(13)] +
    ['rms_mean_var', 'rms_var_var', 'zcr_mean_var', 'zcr_var_var', 'centroid_mean_var',
     'centroid_var_var', 'rolloff_mean_var', 'rolloff_var_var', 'entropy_var', 'num_peaks_var', 'label']
)

df = pd.DataFrame(data, columns=column_names)
df.to_csv("features_robustas_TOTAL.csv", index=False)
print("✅ Características robustas guardadas en 'features_robustas_TOTAL.csv'")
