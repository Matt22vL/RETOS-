import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

def cambiar_velocidad(y, sr, speed_factor=1.0):
    return librosa.effects.time_stretch(y, speed_factor)

def cambiar_tono(y, sr, n_steps=0):
    return librosa.effects.pitch_shift(y, sr, n_steps=n_steps)

def agregar_ruido(y, sr, noise_factor=0.005):
    ruido = np.random.randn(len(y))
    return y + noise_factor * ruido

def cambiar_volumen(y, sr, gain_factor=1.0):
    return y * gain_factor

def extraer_caracteristicas(path):
    y, sr = librosa.load(path, sr=None)
    
    augmentations = [
        lambda y, sr: y,
        lambda y, sr: cambiar_velocidad(y, sr, speed_factor=0.9),
        lambda y, sr: cambiar_velocidad(y, sr, speed_factor=1.1),
        lambda y, sr: cambiar_tono(y, sr, n_steps=2),
        lambda y, sr: cambiar_tono(y, sr, n_steps=-2),
        lambda y, sr: agregar_ruido(y, sr, noise_factor=0.003),
        lambda y, sr: cambiar_volumen(y, sr, gain_factor=1.2),
    ]
    
    features = []
    
    for aug in augmentations:
        y_aug = aug(y, sr)
        
        # Extraemos características
        mfccs = librosa.feature.mfcc(y=y_aug, sr=sr, n_mfcc=13)
        mfccs_mean = mfccs.mean(axis=1)
        
        rms = librosa.feature.rms(y=y_aug).mean()
        zcr = librosa.feature.zero_crossing_rate(y=y_aug).mean()
        centroid = librosa.feature.spectral_centroid(y=y_aug, sr=sr).mean()
        
        feature_vector = np.concatenate([mfccs_mean, [rms, zcr, centroid]])
        features.append(feature_vector)
        
    return np.array(features)

def crear_dataset(ruta_base):
    X = []
    y = []
    carpetas = sorted(os.listdir(ruta_base))
    
    for etiqueta in tqdm(carpetas):
        ruta_carpeta = os.path.join(ruta_base, etiqueta)
        if not os.path.isdir(ruta_carpeta):
            continue
        
        archivos = os.listdir(ruta_carpeta)
        
        for archivo in archivos:
            if archivo.endswith('.wav'):
                ruta_archivo = os.path.join(ruta_carpeta, archivo)
                try:
                    caracteristicas = extraer_caracteristicas(ruta_archivo)
                    X.extend(caracteristicas)
                    y.extend([int(etiqueta)] * len(caracteristicas))
                except Exception as e:
                    print(f"❌ Error con {ruta_archivo}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset listo, shape: {X.shape}")
    return X, y

# Ejemplo de uso
ruta_base = 'C:/Users/semas/OneDrive/Escritorio/RETOS-/Datos/AUD total'
X, y = crear_dataset(ruta_base)
