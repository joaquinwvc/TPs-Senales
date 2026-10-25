import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import warnings
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt

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

#---------------------------------------------------------EJERCICIO 4.1
def suma_modular_am(m1, fc1, m2, fc2, mu):
    t = np.arange(len(m1)) / fs #t es igual para m2
    return ((1 + mu * m1) * np.cos(2 * np.pi * fc1 * t) + (1 + mu * m2) * np.cos(2 * np.pi * fc2 * t))

def ej4_1(m1, m2): #genero audio
    mu = 0.8
    fc1 = 8000
    fc2 = 12000

    w_ = suma_modular_am(m1, fc1, m2, fc2, mu) #modulada

    wavfile.write("w.wav", fs, ( 0.25 * w_).astype(np.float32))
    print(f"Guardado w.wav (max|0.5 s| = {np.max(np.abs(0.25 * w_)):.3f})")

#---------------------------------------------------------EJERCICIO 4.2
def ej4_2(audio): #grafica espectrograma del audio = w = 0.25 * ((1+mu m1) cos fc1 + (1+mu m2) cos fc2)
    nfft = 4096
    ventana = 40 #3840ptos <- (40ms con fs=96000 1/s) 
    f, t, S = espectrograma(audio, fs, ventana, nfft=nfft)
    graficar_espectrograma(f, t, S,
                           f"Espectrograma de w(t) - ventana de {ventana} ms, nfft = {nfft}",
                           f"2/espectrograma_w(t).png", fmax=15000, t_min=0, t_max=10)

#---------------------------------------------------------EJERCICIO 4.3
# Diseño del filtro: Butterworth pasa-bajos de 'orden' polos.
# fs=fs permite dar el corte directamente en Hz.
# output="sos" devuelve secciones de segundo orden (numéricamente estable).
def filtro_pasabajos(v, fs, f_corte=8000, orden=8):
    
    sos = butter(orden, f_corte, btype="low", fs=fs, output="sos")

    # Aplicación del filtro hacia adelante y hacia atrás: fase cero,
    # por lo que la salida queda alineada en el tiempo con la entrada.
    return sosfiltfilt(sos, v)

def _recuperar_mensaje(v, fs, mu, f_corte=8000):
    # 1) Pasa-bajos: queda y(t) = 1/4 [1 + mu m(t)]
    y = filtro_pasabajos(v, fs, f_corte)

    # 2) Se quita la continua (el 1/4 que proviene de la portadora)
    y = y - np.mean(y)

    # 3) Se compensa el factor mu/8 de la amplitud: m_hat ≈ m
    return (8 / mu) * y

#@param w: señal original 
#@param fc: frecuencia de portadora a recuperar
#@param mu: factor de modulación   
def ej4_3(w, mu, fs, m1, m2):
    os.makedirs("3", exist_ok=True)
    #grafico mensajes originales
    nfft = 4096 
    ventana = 40
    f, t, S = espectrograma(m1, fs, ventana, nfft=nfft)
    graficar_espectrograma(f, t, S,
                           f"Espectrograma de m1(t) original - ventana de {ventana} ms, nfft = {nfft}",
                           f"3/espectrograma_m1_original.png", fmax=10000, t_min=0, t_max=10)

    f, t, S = espectrograma(m2, fs, ventana, nfft=nfft)
    graficar_espectrograma(f, t, S,
                           f"Espectrograma de m2(t) original - ventana de {ventana} ms, nfft = {nfft}",
                           f"3/espectrograma_m2_original.png", fmax=10000, t_min=0, t_max=10)


    #Segundo bloque del demodulador: filtro pasa-bajos de Butterworth.
    t = np.arange(len(w)) / fs

    # mensaje en la portadora 1 8000
    v = w * np.cos(2 * np.pi * 8000 * t)
    m1_hat = _recuperar_mensaje(v, fs, mu)

    # mensaje en la portadora 1 12000
    v = w * np.cos(2 * np.pi * 12000 * t)
    m2_hat = _recuperar_mensaje(v, fs, mu)

    f, t, S = espectrograma(m1_hat, fs, ventana)
    graficar_espectrograma(f, t, S,
                           f"Espectrograma de m̂1(t) - ventana de {ventana} ms",
                           f"3/espectrograma_m1hat.png", fmax=10000, t_min=0, t_max=10)

    #Para escucharlas: se normaliza para no saturar
    wavfile.write("3/m1_hat.wav", fs, (0.9 * m1_hat / np.max(np.abs(m1_hat))).astype(np.float32))
    wavfile.write("3/m1_original_normalizado.wav", fs, (0.9 * m1).astype(np.float32))

    f, t, S = espectrograma(m2_hat, fs, ventana)
    graficar_espectrograma(f, t, S,
                           f"Espectrograma de m̂2(t) - ventana de {ventana} ms",
                           f"3/espectrograma_m2hat.png", fmax=10000, t_min=0, t_max=10)

    #Para escucharlas: se normaliza para no saturar
    wavfile.write("3/m2_hat.wav", fs, (0.9 * m2_hat / np.max(np.abs(m2_hat))).astype(np.float32))
    wavfile.write("3/m2_original_normalizado.wav", fs, (0.9 * m2).astype(np.float32))


#---------------------------------------------------------EJERCICIO 4.4

def _ej4_3(w, fc, mu, fs):
    #Segundo bloque del demodulador: filtro pasa-bajos de Butterworth.
    # Señal multiplicada por la portadora local (punto 1)
    t = np.arange(len(w)) / fs
    v = w * np.cos(2 * np.pi * fc * t)

    # Segundo bloque del demodulador
    m_hat = _recuperar_mensaje(v, fs, mu, 2000)

    return m_hat

def ej4_4(m1, m2):
    os.makedirs("4", exist_ok=True)

    #filtro_pasabajos(v, fs, f_corte=8000, orden=8):
    m1_filtrada = filtro_pasabajos(m1, fs, f_corte=2000)
    m2_filtrada = filtro_pasabajos(m2, fs, f_corte=2000)

    #grafico espectrogramas de m1 y m2 filtradas
    nfft = 4096
    ventana = 40 #3840ptos <- (40ms con fs=96000 1/s) 

    f, t, S = espectrograma(m1_filtrada, fs, ventana, nfft=nfft)
    graficar_espectrograma(f, t, S,
                           f"Espectrograma de m1(t) (filtrada) - ventana de {ventana} ms, nfft = {nfft}",
                           f"4/espectrograma_m1_fil.png", fmax=5000, t_min=0, t_max=10)

    f, t, S = espectrograma(m2_filtrada, fs, ventana, nfft=nfft)
    graficar_espectrograma(f, t, S,
                           f"Espectrograma de m2(t) (filtrada) - ventana de {ventana} ms, nfft = {nfft}",
                           f"4/espectrograma_m2_fil.png", fmax=5000, t_min=0, t_max=10)

    mu = 0.8
    fc1 = 8000
    fc2 = 12000
    w_ = suma_modular_am(m1_filtrada, fc1, m2_filtrada, fc2, mu) #modulada

    wavfile.write("w_filtrada.wav", fs, ( 0.25 * w_).astype(np.float32))

    fs_w, w_filtrada = leer_audio("w_filtrada.wav")
    f, t, S = espectrograma(w_filtrada, fs_w, ventana, nfft=nfft)
    graficar_espectrograma(f, t, S,
                           f"Espectrograma de w(t) (filtrada) - ventana de {ventana} ms, nfft = {nfft}",
                           f"4/espectrograma_w_fil.png", fmax=14000, t_min=0, t_max=10)

    m1_hat_fil = _ej4_3( w_filtrada, 8000, 0.8, fs)
    f, t, S = espectrograma(m1_hat_fil, fs, ventana, nfft=nfft)
    graficar_espectrograma(f, t, S,
                           f"Espectrograma de m̂1(t) (filtrada) - ventana de {ventana} ms, nfft = {nfft}",
                           f"4/espectrograma_m1hat_fil.png", fmax=5000, t_min=0, t_max=10)

    #Para escucharlas: se normaliza para no saturar
    wavfile.write("4/m1_hat_fil.wav", fs, (0.9 * m1_hat_fil / np.max(np.abs(m1_hat_fil))).astype(np.float32))
    wavfile.write("4/m1_normalizada_fil.wav", fs, (0.9 * m1).astype(np.float32))

    m2_hat_fil = _ej4_3( w_filtrada, 12000, 0.8, fs)
    f, t, S = espectrograma(m2_hat_fil, fs, ventana, nfft=nfft)
    graficar_espectrograma(f, t, S,
                           f"Espectrograma de m̂2(t) (filtrada) - ventana de {ventana} ms, nfft = {nfft}",
                           f"4/espectrograma_m2hat_fil.png",fmax=5000, t_min=0, t_max=10)

    #Para escucharlas: se normaliza para no saturar
    wavfile.write("4/m2_hat_fil.wav", fs, (0.9 * m2_hat_fil / np.max(np.abs(m2_hat_fil))).astype(np.float32))    
    wavfile.write("4/m2_normalizada_fil.wav", fs, (0.9 * m2).astype(np.float32))



def main():
    
    audios = {}

    for i in ("a.wav", "o.wav"):
        try:
            fs_val, data = leer_audio(i)
        except FileNotFoundError:
            print(f"No se encontró '{i}'. Ponelo en la misma carpeta que el script.")
            continue 
        
        n_canales = 1 if data.ndim == 1 else data.shape[1]
        print(f"Archivo {i} - Canales: {n_canales} ({'mono' if n_canales == 1 else 'estéreo'})")
        print(f"Frecuencia de muestreo: {fs_val} Hz")
        n = obtener_cantidad_muestras(data)
        print(f"Cantidad total de muestras: {n}")

        # Normalizamos la señal en el tiempo (amplitud máxima = 1)
        data_norm = data / np.max(np.abs(data))
        
        audios[i] = data_norm

    # Verificamos que ambos audios se hayan cargado
    if "a.wav" in audios and "o.wav" in audios:
        m1 = audios["a.wav"]
        m2 = audios["o.wav"]

        # Igualar duraciones si fueran distintas (recortando a la menor)
        min_len = min(len(m1), len(m2))
        m1 = m1[:min_len]
        m2 = m2[:min_len]

    # 1. Generar w(t)
    ej4_1(m1, m2)

    # 2. Graficar espectrograma de w(t)
    fs_w, data_w = leer_audio("w.wav")
    # ej4_2(data_w)

    ej4_3(data_w, 0.8, fs, m1, m2)

    # ej4_4(m1, m2)

if __name__ == "__main__":
    main()