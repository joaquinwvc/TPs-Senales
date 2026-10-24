import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import warnings
from numpy.lib.stride_tricks import sliding_window_view


# -------------------------------------------------------------
# Configuración de parámetros
# -------------------------------------------------------------
fs = 96000        # Frecuencia de muestreo en Hz


# 1. Leer el archivo
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


# 3. Obtener la cantidad total de muestras
def obtener_cantidad_muestras(data):
    return data.shape[0]


# 4. Calcular la duración del audio
def calcular_duracion(n_muestras, fs):
    return n_muestras / fs


# Auxiliar: devuelve una lista con un arreglo por canal (1 si es mono, 2 si es estéreo)
def separar_canales(data):
    if data.ndim == 1:
        return [data]
    return [data[:, i] for i in range(data.shape[1])]

# Auxiliar: grafica un tramo de la señal, una figura con un subplot por canal

def normalizar(data):
    m = data.astype(np.float64)
    return m / np.max(np.abs(m))

def _graficar_tramo(data, fs, titulo, archivo_salida, duracion_max=None):
    NOMBRES_CANALES = ["Canal izquierdo", "Canal derecho"]
    canales = separar_canales(data)
    n_tramo = len(canales[0]) if duracion_max is None else min(int(duracion_max * fs), len(canales[0]))
    t = np.arange(n_tramo) / fs

    fig, ejes = plt.subplots(len(canales), 1, figsize=(10, 3 * len(canales)),
                             sharex=True, squeeze=False)
    for i, (canal, ax) in enumerate(zip(canales, ejes[:, 0])):
        ax.plot(t, canal[:n_tramo], linewidth=1)
        ax.set_ylabel("Amplitud")
        ax.grid(True)
        if len(canales) > 1:
            ax.set_title(f"{titulo} - {NOMBRES_CANALES[i] if i < 2 else f'Canal {i + 1}'}")
        else:
            ax.set_title(titulo)
    ejes[-1, 0].set_xlabel("Tiempo [s]")
    fig.tight_layout()

    # Guardar la imagen (crea la carpeta si no existe)
    os.makedirs(os.path.dirname(archivo_salida) or ".", exist_ok=True)
    fig.savefig(archivo_salida, dpi=300)

    plt.show()


# 5. Graficar la señal completa
def graficar_senal_completa(data, fs):
    _graficar_tramo(data, fs, "Señal completa", "img/3_completa.png")

# 6. Obtener el modulo de la fft normalizada de una señal para poder graficar
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

#Grafica el modulo de una fft entre x_min y x_max; y guarda la imagen en la ruta especificada
#@param mag: modulo de la fft
#@param frec_fft: frecuencias de la fft
#@param ruta: ruta donde se guardará la imagen
def graficar_fft(mag, frec, ruta, titulo="|M(f)|", zoom=(-2000, 2000)):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for a, lim in zip(ax, [None, zoom]):
        a.plot(frec, mag)
        a.set_xlabel("Frecuencia [Hz]")
        a.set_ylabel(titulo)
        a.grid(True)
        if lim:
            a.set_xlim(*lim)
    ax[0].set_title(f"{titulo} - espectro completo")
    ax[1].set_title(f"{titulo} - zoom {zoom[0]} a {zoom[1]} Hz")
    plt.tight_layout()
    os.makedirs("img", exist_ok=True)
    plt.savefig(ruta, dpi=150)
    plt.show()

#Espectrograma de m calculado con la FFT de segmentos ventaneados.
#ventana_ms: duración de la ventana en ms.
#nfft: puntos de la FFT (None = largo de la ventana; si es mayor, zero-padding).
#overlap: fracción de solapamiento entre segmentos consecutivos.
#Devuelve f [Hz], t [s] y S (potencia, matriz len(f) x len(t)).
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

def modular_am(m, fc, mu):
    t = np.arange(len(m)) / fs
    return (1 + mu * m) * np.cos(2 * np.pi * fc * t)

#@param data: señal de audio
#@param n: cantidad de muestras de la señal
def ejc2_1(data, n):
    #Obtengo el modulo normalizado de la fft de la señal
    modulo_normalizado_fft, x_fft, x_fft_frec = fft_normalizada(data, n, fs)

    #graico la fft
    graficar_fft(modulo_normalizado_fft, x_fft_frec, "img/1/fft_ejc2_1.png", zoom=(-1000, 1000))

    return modulo_normalizado_fft, x_fft, x_fft_frec


def ejc2_2(m):
    for v in (40, 5):
        f, t, S = espectrograma(m, fs, v)
        graficar_espectrograma(f, t, S, f"Espectrograma de m(t) - ventana de {v} ms",
                               f"img/2/espectrograma_{v}ms.png")

def ejc2_3(m):
    nperseg = int(round(5e-3 * fs))
    for nfft in (nperseg, 4096):
        f, t, S = espectrograma(m, fs, 5, nfft=nfft)
        graficar_espectrograma(f, t, S,
                               f"Espectrograma de m(t) - ventana de 5 ms, nfft = {nfft}",
                               f"img/3/espectrograma_5ms_nfft{nfft}.png")

def ejc2_4(m):
    fc, mu = 8000, 0.8
    s = modular_am(m, fc, mu)
    for v in (40, 5):
        f, t, S = espectrograma(s, fs, v)
        graficar_espectrograma(f, t, S, f"Espectrograma de s(t) - ventana de {v} ms",
                               f"img/4/espectrograma_s_{v}ms.png", fmax= 16000)
    return s

def ejc2_5(s):
    # Punto 5: guardar 0.5 s(t) como WAV
    wavfile.write("s_modulada.wav", fs, (0.5 * s).astype(np.float32))
    print(f"Guardado s_modulada.wav (max|0.5 s| = {np.max(np.abs(0.5 * s)):.3f})")

def main():
    ruta = "ejc2.wav"
    
    try:
            #Leo el archivo de audio
            fs, data = leer_audio(ruta)
    except FileNotFoundError:
            print(f"No se encontró '{ruta}'. Ponelo en la misma carpeta que el script.")
            return
    #ver si es mono o estéreo
    n_canales = 1 if data.ndim == 1 else data.shape[1]
    print(f"Canales: {n_canales} ({'mono' if n_canales == 1 else 'estéreo'})")

    #Muestro por consola la frecuencia de muestreo
    print(f"Frecuencia de muestreo: {fs} Hz")

    #obtengo la cantidad de muestras
    n = obtener_cantidad_muestras(data)
    print(f"Cantidad total de muestras: {n}")

    #Obtengo la duración del audio a a partir de la cant de muestras y la frec de muestro
    duracion = calcular_duracion(n, fs)
    print(f"Duración del audio: {duracion:.3f} s")

    m = normalizar(data)

    #ejercicio 1: Graficar la fft    
    ejc2_1(data, n)

    #ejercicio 2: graficar el espectrograma de ventana 40 ms y 5
    #ejc2_2(m)

    #ejercicio 3: graficar el espectrograma con zero padding
    #ejc2_3(m) #Comentar cuando no se quiera obtener con padding (4096 es bastante pero se nota más :))

    #ejercicio 4: Generar la señal modulada s(t) con fc = 8 kHz y µ = 0,8. Graficar su espectrograma con las dos ventanas anteriores
    s = ejc2_4(m)

    #ejercicio 5: 
    #ejc2_5(s)

if __name__ == "__main__":
    main()