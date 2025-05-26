import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = 'C:\\Users\\semas\\OneDrive\\Escritorio\\RETOS-\\Datos\\AUD prueba\\15\\15try3.wav'
y, sr = librosa.load(audio_path, sr=22050)

# Parámetros del recorte
frame_ms = 50
frame_length = int(frame_ms / 1000 * sr)
hop_length = frame_length

rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

plt.figure(figsize=(10, 4))
plt.plot(times, rms, label="RMS Energy")
plt.axhline(0.005, color='r', linestyle='--', label='Umbral 0.005')
plt.axhline(0.01, color='g', linestyle='--', label='Umbral 0.01')
plt.axhline(0.02, color='orange', linestyle='--', label='Umbral 0.02')
plt.title("Energía RMS del audio")
plt.xlabel("Tiempo (s)")
plt.ylabel("RMS")
plt.legend()
plt.tight_layout()
plt.show()
