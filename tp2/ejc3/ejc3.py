import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import warnings
from numpy.lib.stride_tricks import sliding_window_view

#--------------------------------------------------------------------
# Funciones del ejc
#--------------------------------------------------------------------

# Leer el archivo
def leer_audio(ruta_archivo):
    #fs: frecuencia de muestreo 
    #data: arreglo de las muestras
        # Forma (shape) de data según la cantidad de canales:
        #
        # - Mono: arreglo 1D, con un valor por muestra.
        #   Para tonos de 3 s a 44100 Hz, data.shape es (132300,).
        #
        # - Estéreo: arreglo 2D, con una fila por muestra y una columna por canal.
        #   Por ejemplo, (132300, 2), donde data[:, 0] es el canal izquierdo
        #   y data[:, 1] es el canal derecho.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning) #Audacity warnings, para que no ensucie la consola
        fs, data = wavfile.read(ruta_archivo)
    return fs, data

# Obtener el modulo de la fft normalizada de una señal para poder graficar
def fft_normalizada(señal, n, fs):
    #Normalizo la la señal para que su valor absoluto máximo sea 1.
    x = señal / np.max(np.abs(señal))

    #Calculo la fft de la señal.
    x_fft = np.fft.fft(x)
    x_fft_frec = np.fft.fftfreq(n, d=1/fs)

    # Ordenar de -fs/2 a fs/2 para graficar
    x_fft = np.fft.fftshift(x_fft)
    x_fft_frec = np.fft.fftshift(x_fft_frec)

    modulo_normalizado_fft = np.abs(x_fft) / n

    return modulo_normalizado_fft, x_fft, x_fft_frec

def obtener_cantidad_muestras(data):
    return data.shape[0]

def modular_am(mod_fft, fc, mu, fs):
    t = np.arange(len(mod_fft)) / fs
    return (1 + mu * mod_fft) * np.cos(2 * np.pi * fc * t)

def obtener_vt(s, fc, fs    ):
    t = np.arrange(len(s)) / fs
    return s * np.cos(2 * np.pi * fc * t)

def espectrograma(m, fs, ventana_ms, nfft=None, overlap=0.5):
    nperseg = int(round(ventana_ms * 1e-3 * fs))
    if nfft is None:
        nfft = nperseg
    paso = max(1, nperseg - int(nperseg * overlap))

    w = np.hanning(nperseg)                                 # ventana de Hann
    segmentos = sliding_window_view(m, nperseg)[::paso]     # cada fila = un segmento
    X = np.fft.fft(segmentos * w, n=nfft, axis=1)           # FFT de cada fila
    X = X[:, : nfft // 2 + 1]                               # señal real: solo f >= 0

    S = np.abs(X) ** 2 / (fs * np.sum(w ** 2))              # densidad espectral de potencia
    S[:, 1:-1] *= 2                                         # compensa la mitad descartada

    f = np.abs(np.fft.fftfreq(nfft, d=1 / fs)[: nfft // 2 + 1])
    t = (np.arange(len(segmentos)) * paso + nperseg / 2) / fs   # centro de cada segmento
    return f, t, S.T


def graficar_espectrograma(f, t, S, titulo, ruta, fmin=0, fmax=8000, t_min=None, t_max=None):
    S_db = 10 * np.log10(S + 1e-12)

    # Escala de colores fija según TODA la señal, así los recortes son comparables
    vmax = S_db.max()
    vmin = vmax - 80

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, S_db, shading="gouraud", vmin=vmin, vmax=vmax)
    plt.xlim(t_min, t_max)      # None = límite automático
    plt.ylim(fmin, fmax)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.title(titulo)
    plt.colorbar(label="Potencia [dB]")
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.show()

#Ejc del tp-----------------------------------------------------------------------------------
#3_1:
#obtiene la señal v(t) = s(t) cos(2πfc t) y grafica su espectrograma
#@param s: señal modulada
#@param fc: frecuencia de corte
#@param fs: frecuencia de muestreo
def ejc3_1(s, fc, fs):
     #generar v(t) = s(t) cos(2πfc t)
        v = obtener_vt(s, fc, fs)
    
        #Obtener espectro de v(t)
        vent = 10
        f, t, d = espectrograma(v, fs, vent)
        #gradicar espectro de v(t)
        graficar_espectrograma(f, t, d, f"Espectrograma de v(t) - ventana de {vent} ms",
                                    f"img/2/espectrograma_{vent}ms.png", fmax=2000, t_min=3.5, t_max=5.5)

def main():
    # -------------------------------------------------------------
    # Configuración de parámetros
    # -------------------------------------------------------------
    fs = 96000          # Frecuencia de muestreo en Hz
    ruta = "ejc2.wav"   #ruta del archivo de audio a leer
    fc, mu = 8000, 0.8  #frec de corte y mu de la señla modularizada: s(t) = 1 + µ m(t) cos(2πfc t)
    
    try:
            #Leo el archivo de audio
            fs, data = leer_audio(ruta)
    except FileNotFoundError:
            print(f"No se encontró '{ruta}'. Ponelo en la misma carpeta que el script.")
            return

    #obtengo la cantidad de muestras
    n = obtener_cantidad_muestras(data)
    #calculo fft normalizada
    modulo_normalizado_fft, x_fft, x_fft_frec = fft_normalizada(data, n, fs)
    #modulo la señal
    s = modular_am(modulo_normalizado_fft, fc, mu, fs)


    #Ejc-------------------------------------------------------------------
    ejc3_1(s, fc, fs)
    

if __name__ == "main":
    main()