import os
import sounddevice as sd
import wave
import numpy as np
import joblib
from colorama import Fore, Style
import librosa
import time

# === Configuración ===
DURACION_GRABACION = 8  # Duración de la grabación en segundos
DURACION_ANALISIS = 3   # Duración de audio a analizar (en segundos)
FRECUENCIA_MUESTREO = 22050  # Frecuencia de muestreo
CLASES_VALIDAS = np.array([5, 10, 15, 20, 25])  # Clases válidas
CARPETA_SALIDA = "grabaciones"  # Carpeta para guardar audios

# Crear carpeta de salida si no existe
if not os.path.exists(CARPETA_SALIDA):
    os.makedirs(CARPETA_SALIDA)

# === Función para grabar audio ===
def grabar_audio(nombre_archivo):
    print(Fore.GREEN + f"🎙️ Grabando: {nombre_archivo}..." + Style.RESET_ALL)
    audio = sd.rec(int(DURACION_GRABACION * FRECUENCIA_MUESTREO), samplerate=FRECUENCIA_MUESTREO, channels=1, dtype='int16')
    sd.wait()  # Esperar a que termine la grabación
    with wave.open(nombre_archivo, 'wb') as archivo_wav:
        archivo_wav.setnchannels(1)
        archivo_wav.setsampwidth(2)  # 16 bits = 2 bytes
        archivo_wav.setframerate(FRECUENCIA_MUESTREO)
        archivo_wav.writeframes(audio.tobytes())
    print(Fore.RED + f"🛑 Grabación finalizada: {nombre_archivo}" + Style.RESET_ALL)

# === Función para extraer características de los 3 segundos centrales ===
def extraer_caracteristicas_centrales(audio_path):
    y, sr = librosa.load(audio_path, sr=FRECUENCIA_MUESTREO)
    N_SAMPLES = DURACION_ANALISIS * FRECUENCIA_MUESTREO

    # Extraer los 3 segundos centrales
    inicio = (len(y) - N_SAMPLES) // 2
    y_central = y[inicio:inicio + N_SAMPLES]

    # Extraer características
    mfccs = librosa.feature.mfcc(y=y_central, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    rms = np.mean(librosa.feature.rms(y=y_central))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y_central))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y_central, sr=sr))

    features = list(mfccs_mean) + [rms, zcr, centroid]
    return np.array(features).reshape(1, -1)

# === Función para predecir con el modelo ===
def predecir_audio(audio_path, modelo_path='modelo_randomforest_mentas_TOTAL.pkl', scaler_path='scaler_mentas_TOTAL.pkl'):
    modelo = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)

    # Extraer características
    features = extraer_caracteristicas_centrales(audio_path)
    features_scaled = scaler.transform(features)

    # Realizar predicción
    probs = modelo.predict_proba(features_scaled)[0]
    clases = modelo.classes_
    prediccion = clases[np.argmax(probs)]
    probabilidad_max = np.max(probs)

    print(f"🔊 Predicción: {prediccion} mentas (probabilidad: {probabilidad_max:.2f})")
    return prediccion

# === Función principal para grabar y predecir ===
def grabar_y_predecir():
    print(Fore.YELLOW + "🚦 Preparándote para grabar 5 audios..." + Style.RESET_ALL)
    time.sleep(2)  # Pausa inicial
    predicciones = []

    for i in range(1, 6):  # Grabar 5 audios
        print(Fore.YELLOW + f"⏳ Prepárate para grabar el audio {i}..." + Style.RESET_ALL)
        time.sleep(2)  # Tiempo para prepararse
        nombre_archivo = os.path.join(CARPETA_SALIDA, f"audio_{i}.wav")
        grabar_audio(nombre_archivo)

        # Predecir para el audio grabado
        prediccion = predecir_audio(nombre_archivo)
        predicciones.append(prediccion)

        if i < 5:
            print(Fore.CYAN + "🔄 Descanso antes de la siguiente grabación..." + Style.RESET_ALL)
            time.sleep(2)  # Descanso entre grabaciones

    # Calcular promedio ponderado
    pred_final = CLASES_VALIDAS[np.argmin(np.abs(CLASES_VALIDAS - np.mean(predicciones)))]
    print(Fore.GREEN + f"\n✅ Predicción final (promedio): {pred_final} mentas" + Style.RESET_ALL)

# === USO DIRECTO ===
if __name__ == '__main__':
    grabar_y_predecir()