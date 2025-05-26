import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.fftpack import fft
from scipy.signal import find_peaks

# Función para convertir m4a a wav
def convert_m4a_to_wav(m4a_path):
    wav_path = m4a_path.replace(".m4a", ".wav")
    if not os.path.exists(wav_path):  # Evitar convertir si ya existe
        os.system(f'ffmpeg -i "{m4a_path}" -acodec pcm_s16le -ar 44100 "{wav_path}"')
    return wav_path

# Función para calcular la frecuencia pico media de un archivo de audio
def compute_average_peak_frequency(audio_data, sample_rate, window_size=4096, overlap=0.5):
    step_size = int(window_size * (1 - overlap))
    peak_frequencies = []
    
    for i in range(0, len(audio_data) - window_size, step_size):
        segment = audio_data[i:i + window_size]
        fft_result = np.abs(fft(segment))[:window_size // 2]
        freqs = np.fft.fftfreq(window_size, d=1/sample_rate)[:window_size // 2]
        
        peak_indices, _ = find_peaks(fft_result, height=np.max(fft_result) * 0.2)
        if peak_indices.size > 0:
            peak_frequencies.append(freqs[peak_indices[0]])
    
    return np.mean(peak_frequencies) if peak_frequencies else 0

# Ruta de la carpeta principal
main_directory = input("Ingrese la ruta de la carpeta principal: ").strip()
main_directory = main_directory.replace("\\", "/")  # Corrige problemas en Windows

# Diccionario para almacenar la frecuencia media de cada subcarpeta
folder_frequencies = {}

# Recorrer cada subcarpeta dentro de la carpeta principal
for folder_name in sorted(os.listdir(main_directory)):
    folder_path = os.path.join(main_directory, folder_name)
    if os.path.isdir(folder_path):  # Asegurar que es una carpeta
        total_frequency = []
        
        # Procesar archivos de audio dentro de la subcarpeta
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith(".m4a"):
                file_path = convert_m4a_to_wav(file_path)
            
            if file.endswith(".wav"):
                sample_rate, audio_data = wav.read(file_path)
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=1)  # Convertir a mono si es necesario
                
                avg_freq = compute_average_peak_frequency(audio_data, sample_rate)
                if avg_freq > 0:
                    total_frequency.append(avg_freq)
        
        # Calcular frecuencia media de la carpeta
        folder_frequencies[folder_name] = np.mean(total_frequency) if total_frequency else 0

# Ordenar por número de carpeta
folder_numbers = sorted(int(folder) for folder in folder_frequencies.keys())
folder_means = [folder_frequencies[str(num)] for num in folder_numbers]

# Graficar
plt.figure(figsize=(10, 5))
plt.plot(folder_numbers, folder_means, marker='o', linestyle='-', color='b')
plt.xlabel("Número de mentas")
plt.ylabel("Frecuencia Media (Hz)")
plt.title("frecuencia pico media Vs numero de mentas")
plt.grid(True)
plt.show()
