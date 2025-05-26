import os
import librosa
import numpy as np
import joblib
from scipy.stats import entropy
from scipy.signal import find_peaks

# === Clases válidas
CLASES_VALIDAS = np.array([5, 10, 15, 20, 25])

# === Parámetros
TARGET_SR = 22050
FRAME_SIZE = int(0.2 * TARGET_SR)  # 200 ms
HOP_SIZE = FRAME_SIZE
N_MFCC = 13
MIN_DURATION = 0.01
MAX_DURATION = 3.0
ENERGY_THRESHOLD = 0.003  # Puedes ajustarlo

# === Recorte de golpes por energía ===
def recortar_a_golpes(y, sr, frame_ms=50, energy_threshold=ENERGY_THRESHOLD):
    frame_length = int(frame_ms / 1000 * sr)
    hop_length = frame_length
    energies = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    active_frames = []

    for i, energy in enumerate(energies):
        if energy > energy_threshold:
            start = i * hop_length
            end = start + frame_length
            active_frames.append(y[start:end])

    return np.concatenate(active_frames) if active_frames else y

# === Extracción de características (idéntica a la de entrenamiento)
def extraer_caracteristicas_ventaneadas(audio_path):
    y, sr = librosa.load(audio_path, sr=TARGET_SR)
    y = recortar_a_golpes(y, sr)

    dur = librosa.get_duration(y=y, sr=sr)
    if dur < MIN_DURATION:
        raise ValueError("Audio muy corto (< 0.8s)")
    elif dur > MAX_DURATION:
        y = y[:int(MAX_DURATION * sr)]
    elif dur < MAX_DURATION:
        y = np.pad(y, (0, int(MAX_DURATION * sr) - len(y)))

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
        peaks, _ = find_peaks(env, height=np.max(env) * 0.3)
        num_peaks = len(peaks)

        caracteristicas.append(
            list(mfccs_mean) +
            [rms_mean, rms_var, zcr_mean, zcr_var, centroid_mean, centroid_var,
             rolloff_mean, rolloff_var, entropia, num_peaks]
        )

    caracteristicas = np.array(caracteristicas)
    media = np.mean(caracteristicas, axis=0)
    varianza = np.var(caracteristicas, axis=0)
    return np.concatenate([media, varianza]).reshape(1, -1)

# === Predicción principal
def predecir_promedio_desde_carpeta(carpeta, modelo_path='modelo_randomforest_robusto1.pkl', scaler_path='scaler_robusto1.pkl'):
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
            features = extraer_caracteristicas_ventaneadas(archivo)
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

# === Uso
if __name__ == '__main__':
    carpeta_5_audios = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD prueba\\25'
    predecir_promedio_desde_carpeta(carpeta_5_audios)

