import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import warnings
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt

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

def normalizar(data):
    return data / np.max(np.abs(data))

# Obtener el modulo de la fft normalizada de una señal para poder graficar
def fft_normalizada(señal, n, fs):
    #Normalizo la la señal para que su valor absoluto máximo sea 1.
    x = normalizar(señal)

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

def modular_am(m, fc, mu, fs):
    t = np.arange(len(m)) / fs
    return (1 + mu * m) * np.cos(2 * np.pi * fc * t)

def obtener_vt(s, fc, fs    ):
    t = np.arange(len(s)) / fs
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

# Diseño del filtro: Butterworth pasa-bajos de 'orden' polos.
# fs=fs permite dar el corte directamente en Hz.
# output="sos" devuelve secciones de segundo orden (numéricamente estable).
def filtro_pasabajos(v, fs, f_corte=8000, orden=8):
    
    sos = butter(orden, f_corte, btype="low", fs=fs, output="sos")

    # Aplicación del filtro hacia adelante y hacia atrás: fase cero,
    # por lo que la salida queda alineada en el tiempo con la entrada.
    return sosfiltfilt(sos, v)

def recuperar_mensaje(v, fs, mu, f_corte=8000):
    # 1) Pasa-bajos: queda y(t) = 1/2 [1 + mu m(t)]
    y = filtro_pasabajos(v, fs, f_corte)

    # 2) Se quita la continua (el 1/2 que proviene de la portadora)
    y = y - np.mean(y)

    # 3) Se compensa el factor mu/2 de la amplitud: m_hat ≈ m
    return (2 / mu) * y


#Ejc del tp-----------------------------------------------------------------------------------
#3_1:
#obtiene la señal v(t) = s(t) cos(2πfc t) y grafica su espectrograma
#@param s: señal modulada
#@param fc: frecuencia de corte
#@param fs: frecuencia de muestreo
def ejc3_1(s, fc, fs):
        os.makedirs("img/1", exist_ok=True)
     #generar v(t) = s(t) cos(2πfc t)
        v = obtener_vt(s, fc, fs)
    
        #Obtener espectro de v(t)
        for vent in (40, 5):
            f, t, d = espectrograma(v, fs, vent)
            #gradicar espectro de v(t)
            graficar_espectrograma(f, t, d, f"Espectrograma de v(t) - ventana de {vent} ms",
                                        f"img/1/espectrograma_{vent}ms.png", fmax=24000)
    #guardar v(t) como WAV
        wavfile.write("v.wav", fs, v.astype(np.float32))
        print(f"Guardado v.wav (max|v| = {np.max(np.abs(v)):.3f})")
#3_2:
#Implementa el filtro y compara m_hat con en el tiempo
# y en el espectrograma
#@param m: señal original
#@param s: señal modulada
#@param fc: frecuencia de corte
#@param mu: factor de modulación   
def ejc3_2(m, s, fc, mu, fs):
    #Segundo bloque del demodulador: filtro pasa-bajos de Butterworth.
    #Conserva la copia de la voz en banda base y elimina la réplica
    #centrada en 2*fc = 16 kHz.
    os.makedirs("img/2", exist_ok=True)

    # Señal multiplicada por la portadora local (punto 1)
    t = np.arange(len(s)) / fs
    v = s * np.cos(2 * np.pi * fc * t)

    # Segundo bloque del demodulador
    m_hat = recuperar_mensaje(v, fs, mu)

    # --- Comparación en el tiempo (tramo de 50 ms) ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, m, label="m(t)")
    plt.plot(t, m_hat, label="m̂(t)", alpha=0.7)
    plt.xlim(4.0, 4.05)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("img/2/comparacion_m_mhat.png", dpi=150)
    plt.show()

    #Comparación en el espectrograma
    for ventana in (40, 5):
        f, tt, S = espectrograma(m_hat, fs, ventana)
        graficar_espectrograma(f, tt, S,
                               f"Espectrograma de m̂(t) - ventana de {ventana} ms",
                               f"img/2/espectrograma_mhat_{ventana}ms.png",
                               fmax=8000)

    #Para escucharlas: se normaliza para no saturar
    wavfile.write("m_hat.wav", fs, (0.9 * m_hat / np.max(np.abs(m_hat))).astype(np.float32))
    wavfile.write("m_normalizada.wav", fs, (0.9 * m).astype(np.float32))

    return m_hat

#Demodula con portadora local cos(2*pi*fc*t + phi), mide max|m_hat|
# y grafica el espectrograma para cada fase.

def ejc3_3(m, s, fc, mu, fs, fases_deg=(0, 60, 90)):

    os.makedirs("img/3", exist_ok=True)
    t = np.arange(len(s)) / fs

    resultados = []
    for phi_deg in fases_deg:
        phi = np.deg2rad(phi_deg)

        # Multiplicación por la portadora local desfasada
        v = s * np.cos(2 * np.pi * fc * t + phi)

        # Mismo pasa-bajos y mismo ajuste que en el punto 2
        m_hat = recuperar_mensaje(v, fs, mu)

        # Valor absoluto máximo, descartando 50 ms en cada borde (transitorios del filtro)
        borde = int(0.05 * fs)
        maximo = np.max(np.abs(m_hat[borde:-borde]))
        resultados.append((phi_deg, maximo, np.cos(phi)))
        print(f"phi = {phi_deg:2d}°: max|m_hat| = {maximo:.4f}")

        # Espectrograma (ventana de 40 ms)
        f, tt, S = espectrograma(m_hat, fs, 40)
        graficar_espectrograma(f, tt, S,
                               f"Espectrograma de m̂(t), φ = {phi_deg}°",
                               f"img/3/espectrograma_mhat_phi{phi_deg}.png",
                               fmax=8000)

        # Para escuchar sin normalizar (así se nota la caída de volumen)
        wavfile.write(f"m_hat_phi{phi_deg}.wav", fs, m_hat.astype(np.float32))
    
    return resultados

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
    #normalizo la señal 
    m = normalizar(data)
    #modulo la señal
    s = modular_am(m, fc, mu, fs)


    #Ejc-------------------------------------------------------------------
    ejc3_1(s, fc, fs)
    m_hat = ejc3_2(m, s, fc, mu, fs)
    ejc3_3(m, s, fc, mu, fs)

    

if __name__ == "__main__":
    main()