import librosa
import numpy as np
import pandas as pd
import joblib
import tkinter as tk
from tkinter import filedialog

def extraer_caracteristicas_enriquecidas(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    N_SAMPLES = sr * 3
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]

    # 1) MFCCs
    mfccs_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    # 2) RMS, ZCR, Centroid, Bandwidth, Rolloff
    rms       = np.mean(librosa.feature.rms(y=y))
    zcr       = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid  = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff   = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # 3) Chroma (12)
    chroma_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

    # 4) Tonnetz (6)
    tonnetz_mean = np.mean(
        librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr),
        axis=1
    )

    # 5) Mel spectrogram (tomamos 20 bandas)
    mel_mean = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)[:20]

    # concatenar todo
    features = np.concatenate([  
        mfccs_mean,
        [rms, zcr, centroid, bandwidth, rolloff],
        chroma_mean,
        tonnetz_mean,
        mel_mean
    ])
    return features.reshape(1, -1)


def cuantas_mentas(audio_path,
                   modelo_path='modelo_xgboost_mentas_enriquecido.pkl',
                   scaler_path='scaler_enriquecido_xgb.pkl'):
    try:
        # Cargar el modelo XGBoost entrenado y el scaler
        modelo = joblib.load(modelo_path)
        scaler = joblib.load(scaler_path)

        # Extraer las características del audio
        feats = extraer_caracteristicas_enriquecidas(audio_path)

        # Construir los nombres de las columnas exactamente como en el CSV de entrenamiento
        columnas = (
            [f"mfcc_{i+1}" for i in range(13)] +
            ['rms', 'zcr', 'centroid', 'bandwidth', 'rolloff'] +
            [f"chroma_{i+1}" for i in range(12)] +
            [f"tonnetz_{i+1}" for i in range(6)] +
            [f"mel_{i+1}" for i in range(20)]
        )

        # Crear el DataFrame con las características extraídas
        feats_df = pd.DataFrame(feats, columns=columnas)

        # Escalar las características y hacer la predicción
        feats_scaled = scaler.transform(feats_df)
        probs = modelo.predict_proba(feats_scaled)[0]
        classes = modelo.classes_

        # Obtener las 3 principales predicciones
        idxs = np.argsort(probs)[::-1][:3]
        top_clases = classes[idxs]
        top_probs = probs[idxs]

        print(f"\n🔊 Predicción: {top_clases[0]} mentas")
        print("\n🔎 Top 3 predicciones:")
        for cls, p in zip(top_clases, top_probs):
            print(f"  {cls} mentas: {p*100:.2f}%")

        return top_clases[0]

    except Exception as e:
        print(f"❌ Error al predecir: {e}")
        return None

def seleccionar_archivo():
    # Abrir cuadro de diálogo para seleccionar archivo de audio
    root = tk.Tk()
    root.withdraw()  # No mostrar la ventana principal
    archivo_audio = filedialog.askopenfilename(title="Seleccionar archivo de audio", filetypes=[("Archivos WAV", "*.wav")])
    return archivo_audio

if __name__ == '__main__':
    # Selección del archivo de audio
    ruta_audio = seleccionar_archivo()
    
    if ruta_audio:
        cuantas_mentas(ruta_audio)
