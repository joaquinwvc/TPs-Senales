import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


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
import os

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


# 6. Graficar únicamente los primeros 50 ms
def graficar_primeros_50ms(data, fs):
    _graficar_tramo(data, fs, "Primeros 50 ms", "img/3_completa.png", duracion_max=0.05)


# 7. Graficar únicamente los primeros 500 ms
def graficar_primeros_500ms(data, fs):
    _graficar_tramo(data, fs, "Primeros 500 ms", "img/3_completa.png", duracion_max=0.5)

def ejc3():
    ruta = "Windows_XP_Startup.wav"

    try:
        #1: Leo el archivo de audio
        fs, data = leer_audio(ruta)

    except FileNotFoundError:
        print(f"No se encontró '{ruta}'. Ponelo en la misma carpeta que el script.")
        return

    #ver si es mono o estéreo
    n_canales = 1 if data.ndim == 1 else data.shape[1]
    print(f"Canales: {n_canales} ({'mono' if n_canales == 1 else 'estéreo'})")

    #2: muestro por consola la frecuencia de muestreo
    print(f"Frecuencia de muestreo: {fs} Hz")

    #3: obtengo la cantidad de muestras
    n = obtener_cantidad_muestras(data)
    print(f"Cantidad total de muestras: {n}")

    #4: Obtengo la duración del audio a a partir de la cant de muestras y la frec de muestro
    duracion = calcular_duracion(n, fs)
    print(f"Duración del audio: {duracion:.3f} s")

    #graficos de la señal
    graficar_senal_completa(data, fs, )
    graficar_primeros_50ms(data, fs)
    graficar_primeros_500ms(data, fs)

def main():
    ejc3()

    


if __name__ == "__main__":
    main()