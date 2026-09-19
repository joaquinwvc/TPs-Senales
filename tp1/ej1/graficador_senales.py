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

# -------------------------------------------------------------
# Definición de señales 
# -------------------------------------------------------------
x1 = np.cos(2 * np.pi * 5 * t)
x2 = 2 * np.cos(2 * np.pi * 5 * t)
x3 = np.cos(2 * np.pi * 10 * t)
x4 = np.cos(2 * np.pi * 5 * t + np.pi / 2)

# Lista de pares (señal, título en formato LaTeX, nombre imagen generada)
senales = [
    (x1, "$x_1(t) = \\cos(2 \\pi 5 \\cdot t)$", "1_x1"),
    (x2, "$x_2(t) = 2 \\cos(2 \\pi 5 \\cdot t)$", "1_x2"),
    (x3, "$x_3(t) = \\cos(2 \\pi 10 \\cdot t)$", "1_x3"),
    (x4, "$x_4(t) = \\cos(2 \\pi 5 \\cdot t + \\pi/2)$", "1_x3")
]

# -------------------------------------------------------------
# Generación de un gráfico a la vez
# -------------------------------------------------------------
for x, titulo, nombre in senales:
    graficar_senal(t, x, titulo, nombre)