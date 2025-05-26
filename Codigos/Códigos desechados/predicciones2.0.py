import librosa
import numpy as np
import pandas as pd
import joblib

def extraer_caracteristicas(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)  # Aseguramos misma frecuencia que entrenamiento

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

def cuantas_mentas(audio_path, modelo_path='modelo_randomforest_mentas2.pkl', scaler_path='scaler_mentas2.pkl'):
    try:
        # Cargar modelo y scaler
        modelo = joblib.load(modelo_path)
        scaler = joblib.load(scaler_path)

        # Extraer características del nuevo audio
        features = extraer_caracteristicas(audio_path)

        # Escalar características
        features_scaled = scaler.transform(features)

        # Obtener probabilidades
        probs = modelo.predict_proba(features_scaled)[0]
        clases = modelo.classes_

        # Obtener top 3 predicciones
        top_indices = np.argsort(probs)[::-1][:3]
        top_clases = clases[top_indices]
        top_probs = probs[top_indices]

        # Mostrar predicción principal (la más probable)
        prediccion = top_clases[0]
        print(f"🔊 Predicción: {prediccion} mentas")

        # Mostrar top 3 con probabilidades
        print("\n🔎 Top 3 predicciones:")
        for clase, prob in zip(top_clases, top_probs):
            print(f"   {clase} mentas: {prob * 100:.2f}%")

        return prediccion

    except Exception as e:
        print(f"❌ Error al predecir: {e}")

# === USO DIRECTO ===
if __name__ == '__main__':
    ruta_audio = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD prueba\\20\\20try1.wav'
    cuantas_mentas(ruta_audio)
