import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Rutas
AUDIO_DIR = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Codigos\\METRONOMO\\AUD C'  # <-- cambia esto
OUTPUT_DIR = 'espectrogramas_png'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Iterar por carpetas (1 a 29)
for label in tqdm(os.listdir(AUDIO_DIR), desc="Procesando carpetas"):
    carpeta = os.path.join(AUDIO_DIR, label)
    if not os.path.isdir(carpeta): continue

    for archivo in os.listdir(carpeta):
        if archivo.endswith('.wav'):
            ruta_audio = os.path.join(carpeta, archivo)

            try:
                y, sr = librosa.load(ruta_audio, sr=None)

                # Crear figura sin ejes ni bordes
                plt.figure(figsize=(3, 3))
                plt.axis('off')

                # Crear espectrograma de mel
                S = librosa.feature.melspectrogram(y=y, sr=sr)
                S_dB = librosa.power_to_db(S, ref=np.max)
                librosa.display.specshow(S_dB, sr=sr, cmap='magma')

                # Guardar imagen
                nombre_salida = f"{label}_{archivo.replace('.wav', '')}.png"
                ruta_salida = os.path.join(OUTPUT_DIR, nombre_salida)
                plt.savefig(ruta_salida, bbox_inches='tight', pad_inches=0)
                plt.close()

            except Exception as e:
                print(f"❌ Error con {ruta_audio}: {e}")

print("✅ Espectrogramas guardados en:", OUTPUT_DIR)
