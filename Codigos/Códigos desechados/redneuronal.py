import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Parámetros
TARGET_SR = 22050
MIN_DURATION = 0.8  # segundos
MAX_DURATION = 3.0  # segundos
FRAME_SIZE = int(0.2 * TARGET_SR)  # 200 ms
HOP_SIZE = FRAME_SIZE  # sin superposición
N_MFCC = 13

# Función de extracción de características
def extraer_caracteristicas(audio_path):
    y, sr = librosa.load(audio_path, sr=TARGET_SR)
    dur = librosa.get_duration(y=y, sr=sr)

    if dur < MIN_DURATION:
        raise ValueError("Audio demasiado corto (< 0.8s)")
    elif dur > MAX_DURATION:
        y = y[:int(MAX_DURATION * sr)]
    elif dur < MAX_DURATION:
        y = np.pad(y, (0, int(MAX_DURATION * sr) - len(y)))

    # Extracción de características
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfccs_mean = np.mean(mfccs, axis=1)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_mean = np.mean(zcr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = np.mean(centroid)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = np.mean(rolloff)

    # Concatenar características
    features = np.hstack([mfccs_mean, rms_mean, zcr_mean, centroid_mean, rolloff_mean])
    return features

# Función para cargar los datos de las carpetas
def cargar_datos_y_etiquetas(directorio):
    X = []
    y = []
    for folder_name in os.listdir(directorio):
        folder_path = os.path.join(directorio, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        features = extraer_caracteristicas(file_path)
                        X.append(features)
                        y.append(int(folder_name))  # La etiqueta es el nombre de la carpeta (1-29)
                    except Exception as e:
                        print(f"Error procesando {file_path}: {e}")
    return np.array(X), np.array(y)

# Cargar los datos
directorio = "C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD total"
X, y = cargar_datos_y_etiquetas(directorio)

# Normalización de las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Construcción de la red neuronal
model = models.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(29, activation='softmax')  # 29 clases (1-29 mentas)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento de la red neuronal
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluación del modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en test: {test_accuracy * 100:.2f}%")

# Guardar el modelo entrenado
model.save('modelo_red_neuronal.h5')
