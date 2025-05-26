import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from scipy.signal import find_peaks

# === Parámetros ===22050 
DATASET_PATH = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD total'  # Cambia esto según tu ruta
TARGET_SR = 20000
FRAME_MS = 200 # tamaño de frame para recorte por energía (en milisegundos)
ENERGY_THRESHOLD = 0.003  # umbral de energía para considerar un golpe
FRAME_SIZE = int(0.4 * TARGET_SR)  # 200 ms para extracción de características
HOP_SIZE = FRAME_SIZE  # sin superposición
N_MFCC = 13

# === Función para recortar partes sin golpes (actividad baja) ===
def recortar_a_golpes(y, sr, frame_ms, energy_threshold):
    frame_length = int(frame_ms / 1000 * sr)
    hop_length = frame_length

    energies = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T

    active_frames = []
    for i, energy in enumerate(energies):
        if energy > energy_threshold:
            start = i * hop_length
            end = start + frame_length
            active_frames.append(y[start:end])

    if len(active_frames) == 0:
        return y  # si no detectó nada, devuelve el original
    return np.concatenate(active_frames)

# === Función para extraer características por ventanas ===
def extraer_caracteristicas_ventaneadas(y, sr):
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
        peaks, _ = find_peaks(env, height=np.max(env)*0.3)
        num_peaks = len(peaks)

        caracteristicas.append(
            list(mfccs_mean) +
            [rms_mean, rms_var, zcr_mean, zcr_var, centroid_mean, centroid_var,
             rolloff_mean, rolloff_var, entropia, num_peaks]
        )

    caracteristicas = np.array(caracteristicas)
    media = np.mean(caracteristicas, axis=0)
    varianza = np.var(caracteristicas, axis=0)
    return np.concatenate([media, varianza])

# === Proceso principal ===
data = []
for label in tqdm(range(1, 30), desc="Procesando audios"):
    folder_path = os.path.join(DATASET_PATH, str(label))
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                y, sr = librosa.load(file_path, sr=TARGET_SR)

                # Recortar partes sin golpes antes de extraer características
                y = recortar_a_golpes(y, sr, frame_ms=FRAME_MS, energy_threshold=ENERGY_THRESHOLD)

                # Duración final ajustada
                if len(y) < FRAME_SIZE * 2:
                    continue  # muy corto tras recorte, descartar

                features = extraer_caracteristicas_ventaneadas(y, sr)
                data.append(np.concatenate([features, [label]]))

            except Exception as e:
                print(f"❌ Error con {file_path}: {e}")

# === Guardar características ===
n_features = 13 + 10  # 13 MFCC + 10 adicionales
column_names = (
    [f'mfcc_{i+1}_mean' for i in range(13)] +
    ['rms_mean', 'rms_var', 'zcr_mean', 'zcr_var', 'centroid_mean', 'centroid_var',
     'rolloff_mean', 'rolloff_var', 'entropy', 'num_peaks'] +
    [f'mfcc_{i+1}_var' for i in range(13)] +
    ['rms_mean_var', 'rms_var_var', 'zcr_mean_var', 'zcr_var_var', 'centroid_mean_var',
     'centroid_var_var', 'rolloff_mean_var', 'rolloff_var_var', 'entropy_var', 'num_peaks_var', 'label']
)

df = pd.DataFrame(data, columns=column_names)
df.to_csv("features_robustas_TOTAL.csv", index=False)
print("✅ Características robustas guardadas en 'features_robustas_TOTAL.csv'")



################################################################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import joblib

# === 1. Cargar nuevo dataset robusto
df = pd.read_csv('features_robustas_TOTAL.csv')

print("Distribución de clases:")
print(df['label'].value_counts().sort_index())

# === 2. Separar características y etiquetas
X = df.drop('label', axis=1)
y = df['label']

# === 3. Estandarizar nuevas características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. División train/test42
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.9, random_state=42, stratify=y)

# === 5. Modelo base y búsqueda de hiperparámetros
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [20, None],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'bootstrap': [True]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# === 6. Evaluación
y_pred = best_model.predict(X_test)

print("\n📊 Mejores parámetros encontrados:")
print(grid_search.best_params_)

print("\n📈 Precisión validación cruzada:")
print(f"{grid_search.best_score_:.3f}")

print("\n✅ Precisión en test:")
accuracy = accuracy_score(y_test, y_pred)
print(f"{accuracy:.3f}")

print("\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# === 7. Guardar modelo y scaler actualizados
joblib.dump(best_model, 'modelo_randomforest_robusto1.pkl')
joblib.dump(scaler, 'scaler_robusto1.pkl')
print("🧠 Modelo y scaler robustos guardados con éxito.")


#########################################################################################################################################################################

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
FRAME_SIZE = int(0.5 * TARGET_SR)  # 200 ms
HOP_SIZE = FRAME_SIZE
N_MFCC = 13
MIN_DURATION = 0.5
MAX_DURATION = 3.0
ENERGY_THRESHOLD = 0.015  # Puedes ajustarlo

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
    carpeta_5_audios = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD prueba\\10\\10_2'
    predecir_promedio_desde_carpeta(carpeta_5_audios)
