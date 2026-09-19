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
x = (1 + np.cos(2 * np.pi * t) / 2) * np.cos(2 * np.pi * 20 * t)
y = np.cos(2 * np.pi * 20 * t)

# Lista de pares (señal, título en formato LaTeX, nombre imagen generada)
senales = [
    (x, "$x(t) = (1 + ~ \\frac{1}{2} ~ \\cos(2 \\pi \\cdot t)) ~ \\cos(2 \\pi 20 \\cdot t)$", "5_x"),
    (y, "$y(t) = \\cos(2 \\pi 20 \\cdot t)$", "5_cos20hz"),
]

# -------------------------------------------------------------
# Generación de un gráfico a la vez
# -------------------------------------------------------------
for x, titulo, nombre in senales:
    graficar_senal(t, x, titulo, nombre)