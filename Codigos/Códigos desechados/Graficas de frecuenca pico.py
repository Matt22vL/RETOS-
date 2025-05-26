import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
import pandas as pd
from scipy.fftpack import fft
from scipy.signal import find_peaks

def convert_m4a_to_wav(m4a_path):
    """Convierte un archivo .m4a a .wav si aún no existe"""
    wav_path = m4a_path.replace(".m4a", ".wav")
    if not os.path.exists(wav_path):
        subprocess.run(["ffmpeg", "-i", m4a_path, "-acodec", "pcm_s16le", "-ar", "44100", wav_path], check=True)
    return wav_path

def compute_peak_history(audio_data, sample_rate, window_size=4096, overlap=0.5):
    """Calcula la historia de frecuencia pico en una señal de audio"""
    step_size = int(window_size * (1 - overlap))
    peak_frequencies = []
    times = []
    
    for i in range(0, len(audio_data) - window_size, step_size):
        segment = audio_data[i:i + window_size]
        fft_result = np.abs(fft(segment))[:window_size // 2]
        freqs = np.fft.fftfreq(window_size, d=1/sample_rate)[:window_size // 2]
        
        peak_indices, _ = find_peaks(fft_result, height=np.max(fft_result) * 0.2)
        peak_freq = freqs[peak_indices[0]] if peak_indices.size > 0 else 0
        
        peak_frequencies.append(peak_freq)
        times.append(i / sample_rate)
    
    return np.array(times), np.array(peak_frequencies)

def process_audio_files(folder_path):
    """Procesa todos los archivos de audio en una carpeta y calcula estadísticas"""
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.m4a', '.wav'))]
    results = {}
    avg_frequencies = []
    
    for file in audio_files:
        file_path = os.path.join(folder_path, file)
        if file.endswith(".m4a"):
            file_path = convert_m4a_to_wav(file_path)
        
        sample_rate, audio_data = wav.read(file_path)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)  # Convertir a mono si es necesario
        
        times, peak_frequencies = compute_peak_history(audio_data, sample_rate)
        avg_freq = np.mean(peak_frequencies)
        avg_frequencies.append(avg_freq)
        
        results[file] = {"times": times, "frequencies": peak_frequencies, "avg_frequency": avg_freq}
    
    overall_avg_freq = np.mean(avg_frequencies)
    return results, overall_avg_freq

# Ejecutar el procesamiento de la carpeta
directory = input("Ingrese la ruta de la carpeta con los archivos de audio: ")
data, overall_avg = process_audio_files(directory)
print(f"Frecuencia media de toda la carpeta: {overall_avg:.2f} Hz")

# Preguntar al usuario qué audios graficar
print("Archivos procesados:")
file_names = list(data.keys())
for i, name in enumerate(file_names):
    print(f"[{i}] {name}")

choice = input("Ingrese los números de los archivos a graficar separados por comas (o 'all' para todos): ")
if choice.lower() == "all":
    selected_files = file_names
else:
    indices = [int(i) for i in choice.split(',') if i.isdigit() and int(i) < len(file_names)]
    selected_files = [file_names[i] for i in indices]

# Graficar los archivos seleccionados
plt.figure(figsize=(10, 5))
for file in selected_files:
    plt.plot(data[file]["times"], data[file]["frequencies"], marker="o", linestyle="-", markersize=2, linewidth=0.8, alpha=0.8, label=file)

plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia Pico (Hz)")
plt.title("Comparación de Picos de Frecuencia en Audios de 1 menta")
#plt.legend()
plt.grid(True)
plt.show()
