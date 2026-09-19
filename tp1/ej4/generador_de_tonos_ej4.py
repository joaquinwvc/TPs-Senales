import numpy as np
from scipy.io import wavfile

def generate_tone(frequency, duration, sample_rate=44100):
    """
    Genera un tono en forma de onda coseno.

    Argumentos:
        frequency (float): Frecuencia del tono en Hz.
        duration (float): Duración del tono en segundos.
        sample_rate (int): Frecuencia de muestreo en Hz (por defecto 44100).
    Returns:
     La señal de audio generada.
    """
    # Vector de tiempo 't'
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Señal x(t) = cos(2 * pi * f * t)
    signal = np.cos(2 * np.pi * frequency * t)
    
    # Normalizar la señal para guardarla como audio de 16 bits
    signal_normalized = np.int16(signal * 32767)
    return signal_normalized

def save_audio(filename, signal, sample_rate=44100):
    """
    Guarda una señal como un archivo de audio WAV.

    Args:
        filename (str): Nombre del archivo a guardar (ej. "audio.wav").
        signal (np.ndarray): La señal de audio.
        sample_rate (int): Frecuencia de muestreo en Hz.
    """
    wavfile.write(filename, sample_rate, signal)
    print(f"Archivo generado exitosamente: {filename}")

def main():
    # Parámetros comunes
    duracion = 3.0  # Al menos 3 segundos como pide el enunciado
    fs = 44100      # Frecuencia de muestreo (CD quality)

    # --- Generación de x1(t) = cos(2*pi*440*t) ---
    frecuencia1 = 440
    print(f"Generando tono de {frecuencia1} Hz...")
    senial1 = generate_tone(frecuencia1, duracion, fs)
    save_audio("x1_tono_440Hz.wav", senial1, fs)

    # --- Generación de x2(t) = cos(2*pi*880*t) ---
    frecuencia2 = 880
    print(f"Generando tono de {frecuencia2} Hz...")
    senial2 = generate_tone(frecuencia2, duracion, fs)
    save_audio("x2_tono_880Hz.wav", senial2, fs)
    
    print("\n¡Proceso finalizado! Los audios están listos para ser escuchados.")

if __name__ == "__main__":
    main()
