import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.fftpack import fft
from scipy.signal import find_peaks

def convert_m4a_to_wav(m4a_path):
    """Convierte un archivo .m4a a .wav si aún no existe"""
    wav_path = m4a_path.replace(".m4a", ".wav")
    if not os.path.exists(wav_path):
        subprocess.run(["ffmpeg", "-i", m4a_path, "-acodec", "pcm_s16le", "-ar", "44100", wav_path], check=True)
    return wav_path

def weighted_moving_average(data, window_size=5):
    """Aplica una media móvil ponderada para suavizar los datos"""
    weights = np.arange(1, window_size + 1)
    smoothed = np.convolve(data, weights / weights.sum(), mode='valid')
    return np.concatenate((data[:window_size-1], smoothed))

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
    
    peak_frequencies = weighted_moving_average(np.array(peak_frequencies))  # Aplicar suavizado
    return np.array(times), peak_frequencies

def process_audio_files(folder_path):
    """Procesa todos los archivos de audio en una carpeta y agrupa por número de mentas"""
    overall_avg_frequencies = {}
    
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        
        # Extraer el número de mentas de la carpeta (asegurando que sea numérico)
        try:
            num_mints = int(subfolder)
        except ValueError:
            print(f"Advertencia: {subfolder} no es un número válido, se ignorará.")
            continue
        
        audio_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.m4a', '.wav'))]
        avg_frequencies = []
        
        for file in audio_files:
            file_path = os.path.join(subfolder_path, file)
            if file.endswith(".m4a"):
                file_path = convert_m4a_to_wav(file_path)
            
            sample_rate, audio_data = wav.read(file_path)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)  # Convertir a mono si es necesario
            
            _, peak_frequencies = compute_peak_history(audio_data, sample_rate)
            avg_freq = np.mean(peak_frequencies)
            avg_frequencies.append(avg_freq)
        
        # Guardar la frecuencia promedio
        overall_avg_frequencies[num_mints] = np.mean(avg_frequencies)
    
    return overall_avg_frequencies

# Ejecutar el procesamiento de la carpeta
directory = input("Ingrese la ruta de la carpeta con los archivos de audio: ")
overall_avg = process_audio_files(directory)

# Pedir el rango de mentas a graficar
try:
    min_mints = int(input("Ingrese el número mínimo de mentas a graficar: "))
    max_mints = int(input("Ingrese el número máximo de mentas a graficar: "))
except ValueError:
    print("Error: Debe ingresar valores numéricos.")
    exit()

# Filtrar los datos según el rango seleccionado
filtered_avg = {m: f for m, f in overall_avg.items() if min_mints <= m <= max_mints}

# Verificar si hay datos en el rango seleccionado
if not filtered_avg:
    print("No hay datos en el rango especificado.")
    exit()

# Extraer datos filtrados para graficar
num_mentas = np.array(list(filtered_avg.keys()))
frecuencia_promedio = np.array(list(filtered_avg.values()))

# Ajuste lineal y cuadrático
coef_linear = np.polyfit(num_mentas, frecuencia_promedio, 1)  # y = ax + b
coef_cuadratico = np.polyfit(num_mentas, frecuencia_promedio, 2)  # y = ax^2 + bx + c

# Generar valores ajustados
mentas_fit = np.linspace(min(num_mentas), max(num_mentas), 100)
frecuencia_linear = np.polyval(coef_linear, mentas_fit)
frecuencia_cuadratica = np.polyval(coef_cuadratico, mentas_fit)

# Graficar
plt.figure(figsize=(8, 5))
plt.scatter(num_mentas, frecuencia_promedio, color='blue', label='Datos observados')
plt.plot(mentas_fit, frecuencia_linear, 'r--', label=f'Ajuste lineal: {coef_linear[0]:.2f}x + {coef_linear[1]:.2f}')
plt.plot(mentas_fit, frecuencia_cuadratica, 'g-', label='Ajuste cuadrático')

plt.xlabel("Número de mentas")
plt.ylabel("Frecuencia promedio (Hz)")
plt.title(f"Relación entre número de mentas y frecuencia promedio ({min_mints}-{max_mints} mentas)")
plt.legend()
plt.grid(True)
plt.show()
