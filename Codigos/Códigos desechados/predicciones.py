import librosa
import numpy as np
import pandas as pd
import joblib

def extraer_caracteristicas(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

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

def cuantas_mentas(audio_path, modelo_path='modelo_randomforest_mentas2.0.pkl'):
    try:
        # Cargar modelo entrenado
        modelo = joblib.load(modelo_path)

        # Extraer características del nuevo audio
        features = extraer_caracteristicas(audio_path)

        # Asignar nombres de columnas para evitar el warning
        columnas = [f"mfcc_{i+1}" for i in range(13)] + ['rms', 'zcr', 'centroid']
        features_df = pd.DataFrame(features, columns=columnas)

        # Hacer la predicción
        prediccion = modelo.predict(features_df)[0]
        print(f"🔊 Predicción: {prediccion} mentas")
        return prediccion

    except Exception as e:
        print(f"❌ Error al predecir: {e}")

# Uso directo
if __name__ == '__main__':
    ruta_audio = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD prueba\\20\\20try5.wav'  
    cuantas_mentas(ruta_audio)


