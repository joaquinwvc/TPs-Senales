import numpy as np
import matplotlib.pyplot as plt

def graficar_senal(t, x, titulo, nombre_archivo, ylabel="Amplitud"):
    plt.figure(figsize=(8, 4))
    plt.plot(t, x, color='b', linewidth=1.5)
    plt.title(titulo, fontsize=12)
    plt.xlabel("Tiempo [s]")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xlim([t[0], t[-1]])
    plt.tight_layout()

    plt.savefig(nombre_archivo, dpi=300)

    plt.show()  # Muestra una figura a la vez

# -------------------------------------------------------------
# Configuración de parámetros
# -------------------------------------------------------------
fs = 1000        # Frecuencia de muestreo en Hz 
duracion = 1.0   # Duración de la señal en segundos

# Vector de tiempo
t = np.arange(0, duracion, 1 / fs)

def ejc2():

    #Definición de señal ejc 2: x(t) = cos(2π5t) + 0,5 cos(2π20t)
    x = np.cos(2*np.pi * 5 * t) + 0.5*np.cos(2 * np.pi * 20 * t)

    #defino el titulo del gráfico
    titulo = r"$x(t) = \cos(2\pi \cdot 5t) + 0.5\cos(2\pi \cdot 20t)$"

    #grafico la señal
    graficar_senal(t, x, titulo, "2_suma_cosenos.png")

def main():
    ejc2()

if __name__ == "__main__":
    main()