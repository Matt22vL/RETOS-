import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
import pandas as pd
import joblib

DURACION = 4  # segundos
FS = 44100    # frecuencia de muestreo

def grabar_audio(nombre_archivo='grabacion.wav'):
    print(f"🎙 Grabando audio por {DURACION} segundos...")
    audio = sd.rec(int(DURACION * FS), samplerate=FS, channels=1, dtype='int16')
    sd.wait()
    wav.write(nombre_archivo, FS, audio)
    print("✅ Grabación completa.")
    return nombre_archivo

def extraer_caracteristicas(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    features = list(mfccs_mean) + [rms, zcr, centroid]
    return np.array(features).reshape(1, -1)

def cuantas_mentas(audio_path, modelo_path='modelo_randomforest_mentas2.pkl'):
    try:
        modelo = joblib.load(modelo_path)
        features = extraer_caracteristicas(audio_path)

        columnas = [f"mfcc_{i+1}" for i in range(13)] + ['rms', 'zcr', 'centroid']
        features_df = pd.DataFrame(features, columns=columnas)

        prediccion = modelo.predict(features_df)[0]
        print(f"🔊 Predicción: {prediccion} mentas")
        return prediccion
    except Exception as e:
        print(f"❌ Error al predecir: {e}")

if __name__ == '__main__':
    archivo = grabar_audio()
    cuantas_mentas(archivo)
