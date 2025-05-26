import os
import librosa
import numpy as np
import joblib

# === Clases válidas permitidas por el sistema
CLASES_VALIDAS = np.array([5, 10, 15, 20, 25])

# === Función para extraer características del audio (sin MFCC)
def extraer_caracteristicas(audio_path):
    TARGET_SR = 22050
    DURACION_SEGUNDOS = 3
    N_SAMPLES = TARGET_SR * DURACION_SEGUNDOS

    y, sr = librosa.load(audio_path, sr=TARGET_SR)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]

    # Características
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=TARGET_SR))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=TARGET_SR))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # Chroma (promedio de los 12 coeficientes)
    chroma = librosa.feature.chroma_stft(y=y, sr=TARGET_SR)
    chroma_mean = np.mean(chroma, axis=1)

    # Espectrograma - solo 10 primeros bins
    S = np.abs(librosa.stft(y))
    spec_mean = np.mean(S, axis=1)[:10]

    # Vector final
    features = [rms, zcr, centroid, rolloff, bandwidth, flatness] + list(chroma_mean) + list(spec_mean)
    return np.array(features).reshape(1, -1)

# === Función principal
def predecir_promedio_desde_carpeta(carpeta, modelo_path='modelo_randomforest_mentas_sinMFCC.pkl', scaler_path='scaler_mentas_sinMFCC.pkl'):
    modelo = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)

    archivos = sorted([os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.lower().endswith('.wav')])

    if len(archivos) != 5:
        print("❌ La carpeta debe contener exactamente 5 archivos de audio .wav")
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
    carpeta_5_audios = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD prueba\\10\\10_4'
    predecir_promedio_desde_carpeta(carpeta_5_audios)
