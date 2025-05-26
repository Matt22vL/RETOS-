import os
import numpy as np
import pandas as pd
import joblib
import librosa
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar modelo y scaler
modelo = joblib.load('modelo_randomforest_mentas_TOTAL.pkl')
scaler = joblib.load('scaler_mentas_TOTAL.pkl')

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
    # Crear DataFrame con nombres de columnas
    feature_names = [f'mfcc_{i+1}' for i in range(13)] + ['rms', 'zcr', 'centroid']
    return pd.DataFrame([features], columns=feature_names)

# Ruta a la carpeta principal
carpeta_base = r'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD P M'

resultados = []

for carpeta in os.listdir(carpeta_base):
    carpeta_path = os.path.join(carpeta_base, carpeta)
    if not os.path.isdir(carpeta_path):
        continue
    # Extraer la etiqueta real del nombre de la carpeta (ej: '5_1' -> 5)
    etiqueta_real = int(carpeta.split('_')[0])
    for archivo in os.listdir(carpeta_path):
        if archivo.lower().endswith('.wav'):
            audio_path = os.path.join(carpeta_path, archivo)
            features_df = extraer_caracteristicas(audio_path)
            features_scaled = scaler.transform(features_df)
            pred = modelo.predict(features_scaled)[0]
            resultados.append({'real': etiqueta_real, 'prediccion': pred})

# Convertir a DataFrame
df = pd.DataFrame(resultados)

# Graficar distribución (boxplot o swarmplot)
plt.figure(figsize=(10,6))
sns.boxplot(x='real', y='prediccion', data=df, showfliers=False)
sns.swarmplot(x='real', y='prediccion', data=df, color='k', alpha=0.5)
plt.title('Distribución de predicciones individuales por cantidad real de mentas')
plt.xlabel('Cantidad real de mentas')
plt.ylabel('Predicción del modelo')
plt.tight_layout()
plt.show()