import os
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Reutilizamos tu función de extracción de características
def extraer_caracteristicas(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)

    N_SAMPLES = 22050 * 3
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

# === Función para evaluar todos los audios en las carpetas
def evaluar_modelo_en_carpeta(carpeta_raiz, modelo_path='modelo_randomforest_mentas2.pkl', scaler_path='scaler_mentas2.pkl'):
    modelo = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)

    resultados = []

    for carpeta in os.listdir(carpeta_raiz):
        carpeta_path = os.path.join(carpeta_raiz, carpeta)
        if not os.path.isdir(carpeta_path):
            continue
        try:
            etiqueta_real = int(carpeta)
        except ValueError:
            continue  # Ignorar carpetas que no tienen número

        for archivo in os.listdir(carpeta_path):
            if not archivo.lower().endswith('.wav'):
                continue

            audio_path = os.path.join(carpeta_path, archivo)
            try:
                features = extraer_caracteristicas(audio_path)
                features_scaled = scaler.transform(features)
                proba = modelo.predict_proba(features_scaled)[0]
                clases = modelo.classes_
                prediccion = clases[np.argmax(proba)]

                resultados.append({
                    'archivo': archivo,
                    'ruta': audio_path,
                    'real': etiqueta_real,
                    'predicho': prediccion,
                    'probabilidad_predicha': np.max(proba)
                })

            except Exception as e:
                print(f"Error procesando {audio_path}: {e}")

    df_resultados = pd.DataFrame(resultados)
    print("\n✅ Resultados:")
    print(df_resultados[['archivo', 'real', 'predicho', 'probabilidad_predicha']])

    acc = accuracy_score(df_resultados['real'], df_resultados['predicho'])
    print(f"\n🎯 Precisión total del modelo: {acc*100:.2f}%")

    cm = confusion_matrix(df_resultados['real'], df_resultados['predicho'], labels=sorted(modelo.classes_))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(modelo.classes_))
    disp.plot(cmap='Blues', xticks_rotation=90)
    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.show()

    return df_resultados

# === USO ===
if __name__ == '__main__':
    carpeta_test = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD prueba'
    evaluar_modelo_en_carpeta(carpeta_test)
