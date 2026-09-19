import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Configuración de parámetros
# -------------------------------------------------------------
fs = 1000        # Frecuencia de muestreo en Hz 
duracion = 1.0   # Duración de la señal en segundos

# Vector de tiempo
t = np.arange(0, duracion, 1 / fs)

#---------------------------------------------------------------
# Funciones----------------------------------------------------- 
#---------------------------------------------------------------
def graficar_senal(t, x, titulo, nombre_archivo, ylabel="Amplitud", decorar=None):
    plt.figure(figsize=(8, 4))
    plt.plot(t, x, color='b', linewidth=1.5)
    plt.title(titulo, fontsize=12)
    plt.xlabel("Tiempo [s]")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xlim([t[0], t[-1]])

    #Permite pasarle una función que modifique el grafico antes de guardarlo
    if decorar is not None:
        decorar()

    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300)
    plt.show()

def resaltar_periodicidad(T, duracion, y_flecha=1.6):
    # Sombreo períodos alternados
    for k in range(int(duracion / T)):
        if k % 2 == 0:
            plt.axvspan(k * T, (k + 1) * T, color='orange', alpha=0.2)
    # Flecha que marca un período
    plt.annotate("", xy=(T, y_flecha), xytext=(0, y_flecha),
                 arrowprops=dict(arrowstyle='<->', color='r', lw=1.5))
    plt.text(T / 2, y_flecha + 0.05, f"T = {T} s", color='r', ha='center')
    plt.ylim(-y_flecha - 0.2, y_flecha + 0.4)

def ejc2():

    #Definición de señal ejc 2: x(t) = cos(2π5t) + 0,5 cos(2π20t)
    x = np.cos(2*np.pi * 5 * t) + 0.5*np.cos(2 * np.pi * 20 * t)

    #defino el titulo del gráfico
    titulo = r"$x(t) = \cos(2\pi \cdot 5t) + 0.5\cos(2\pi \cdot 20t)$"

    #grafico la señal
    
    T = 0.2  # período fundamental [s]

    graficar_senal(t, x, titulo, "2_suma_cosenos.png",
                   decorar=lambda: resaltar_periodicidad(T, duracion))

def main():
    ejc2()

if __name__ == "__main__":
    main()