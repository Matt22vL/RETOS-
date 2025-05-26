import os
import librosa
import numpy as np
import joblib

CLASES_VALIDAS = np.array([5, 10, 15, 20, 25])

def extraer_caracteristicas(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    N_SAMPLES = 22050 * 4
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features = list(mfccs_mean) + [rms, zcr, centroid]
    return np.array(features).reshape(1, -1)

def predecir_promedio_desde_carpeta(carpeta, modelo_path='modelo_randomforest_mentas_TOTAL.pkl', scaler_path='scaler_mentas_TOTAL.pkl'):
    modelo = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)

    archivos = sorted([os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.lower().endswith('.wav')])

    if len(archivos) < 2:
        print("❌ La carpeta debe contener al menos 2 archivos de audio .wav")
        return

    predicciones = []
    probabilidades = []
    print("🔍 Predicciones individuales (ponderadas por probabilidad):")
    for archivo in archivos:
        try:
            features = extraer_caracteristicas(archivo)
            features_scaled = scaler.transform(features)
            probs = modelo.predict_proba(features_scaled)[0]
            clases = modelo.classes_
            prediccion = clases[np.argmax(probs)]
            probabilidad_max = np.max(probs)
            predicciones.append(prediccion)
            probabilidades.append(probabilidad_max)
            print(f"   {os.path.basename(archivo)} → {prediccion} mentas, probabilidad: {probabilidad_max:.2f}")
        except Exception as e:
            print(f"Error procesando {archivo}: {e}")
            return

    weighted_average = np.average(predicciones, weights=probabilidades)
    pred_final = CLASES_VALIDAS[np.argmin(np.abs(CLASES_VALIDAS - weighted_average))]
    print(f"\n📈 Promedio ponderado de predicciones: {weighted_average:.2f}")
    print(f"✅ Predicción final (redondeada a clase válida): {pred_final} mentas")
    return pred_final

# === USO DIRECTO
if __name__ == '__main__':
    # Cambia esta ruta a la carpeta donde descargaste los audios desde Drive
    carpeta_audios = '/storage/emulated/0/Download/audios_mentas'
    predecir_promedio_desde_carpeta(carpeta_audios)