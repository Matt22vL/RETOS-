import librosa
import numpy as np
import pandas as pd
import joblib

def extraer_caracteristicas(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)

    # Asegurar duración de 3 segundos
    N_SAMPLES = 22050 * 3
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Otras características
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Vector final
    features = list(mfccs_mean) + [rms, zcr, centroid]
    return np.array(features).reshape(1, -1)

def cuantas_mentas(audio_path, modelo_path='modelo_randomforest_mentas_TOTAL.pkl', scaler_path='scaler_mentas_TOTAL.pkl'):
    try:
        # Cargar modelo y scaler
        modelo = joblib.load(modelo_path)
        scaler = joblib.load(scaler_path)

        # Extraer características del nuevo audio
        features = extraer_caracteristicas(audio_path)

        # Convertir a DataFrame con nombres para evitar warning
        columnas = [f'mfcc_{i+1}' for i in range(13)] + ['rms', 'zcr', 'centroid']
        features_df = pd.DataFrame(features, columns=columnas)

        # Escalar características
        features_scaled = scaler.transform(features_df)

        # Realizar predicción
        prediccion = modelo.predict(features_scaled)[0]
        prediccion_entera = int(round(prediccion))
        #prediccion_entera = int(round(prediccion / 5) * 5)
        # Mostrar resultados
        print(f"🔊 Predicción (decimal): {prediccion:.2f} mentas")
        print(f"🎯 Predicción redondeada: {prediccion_entera} mentas")

        return prediccion_entera

    except Exception as e:
        print(f"❌ Error al predecir: {e}")

# === USO DIRECTO ===
if __name__ == '__main__':
    ruta_audio = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Codigos\\METRONOMO\AUD P M\\10_1'
    cuantas_mentas(ruta_audio)
