import numpy as np
import matplotlib.pyplot as plt
import os
import sympy as sp
from sympy.calculus.util import periodicity

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
def graficar_senal(t, x, titulo, nombre_archivo, ylabel="Amplitud", decorar=None, carpeta="."):
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

    #permite pasar la direccion de la carpeta en donde se guardara el archivo
    ruta_de_guardado = os.path.join(carpeta, nombre_archivo)
    plt.savefig(ruta_de_guardado, dpi=300)

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

def calculo_analitico_periodo(formula_str):
    """
    Calcula el período exacto recibiendo la fórmula como string.
    """
    # Convierte el string a una expresión matemática de SymPy
    expr = sp.sympify(formula_str)
    
    simbolos = list(expr.free_symbols)
    if not simbolos:
        return None
        
    variable = simbolos[0]
    T = periodicity(expr, variable)
    
    return float(T) if T is not None else None

# --------------------------------------------------------------------------------------
# Ejercicios
# --------------------------------------------------------------------------------------
def ejc1(): 
    # -------------------------------------------------------------
    # Definición de señales 
    # -------------------------------------------------------------
    x1 = np.cos(2 * np.pi * 5 * t)
    x2 = 2 * np.cos(2 * np.pi * 5 * t)
    x3 = np.cos(2 * np.pi * 10 * t)
    x4 = np.cos(2 * np.pi * 5 * t + np.pi / 2)

    # Lista de pares (señal, título en formato LaTeX, nombre imagen generada)
    senales = [
        (x1, "$x_1(t) = \\cos(2 \\pi 5 \\cdot t)$", "1_x1.png"),
        # (x2, "$x_2(t) = 2 \\cos(2 \\pi 5 \\cdot t)$", "1_x2.png"),
        # (x3, "$x_3(t) = \\cos(2 \\pi 10 \\cdot t)$", "1_x3.png"),
        # (x4, "$x_4(t) = \\cos(2 \\pi 5 \\cdot t + \\pi/2)$", "1_x4.png")
    ]

    # -------------------------------------------------------------
    # Generación de un gráfico a la vez
    # -------------------------------------------------------------
    for x, titulo, nombre in senales:
        graficar_senal(t, x, titulo, nombre)

def ejc2():

    #Definición de señal ejc 2: x(t) = cos(2π5t) + 0,5 cos(2π20t)
    x = np.cos(2*np.pi * 5 * t) + 0.5*np.cos(2 * np.pi * 20 * t)

    #defino el titulo del gráfico
    titulo = r"$x(t) = \cos(2\pi \cdot 5t) + 0.5\cos(2\pi \cdot 20t)$"

    #grafico la señal
    
    T = 0.2  # período fundamental [s]

    graficar_senal(t, x, titulo, "2_suma_cosenos.png",
                   decorar=lambda: resaltar_periodicidad(T, duracion))

def ejc4():
    fs = 40000
    duracion_x1 = 0.006
    duracion_x2 = 0.006

    t_x1 = np.arange(0, duracion_x1, 1 / fs)
    t_x2 = np.arange(0, duracion_x2, 1 / fs)

    x1 = np.cos(2 * np.pi * 440 * t_x1)
    x2 = np.cos(2 * np.pi * 880 * t_x2)

    graficar_senal(t_x1, x1, "$\\cos(2 \\pi 440 \\cdot t)$", "4_x1_440hz.png", decorar=lambda:resaltar_periodicidad(0.0023, duracion_x1, y_flecha = 1.1), carpeta = "ej4")
    graficar_senal(t_x2, x2, "$cos(2 \\pi 880 \\cdot t)$", "4_x1_880hz.png", decorar=lambda:resaltar_periodicidad(0.0011, duracion_x1, y_flecha = 1.1), carpeta = "ej4")



def ejc5():
    duracion_x = 5   # Duración de la señal en segundos

    periodo_x = calculo_analitico_periodo("(1+0.5*cos(2*pi*t))*cos(2*pi*20*t)")  # Duración del periodo de x
    duracion_periodo_x = periodo_x
    
    duracion_y = 0.15   # Duración de la señal en segundos

    duracion_g = 3

    # Vector de tiempo
    t_x = np.arange(0, duracion_x, 1 / fs)
    t_periodo_x = np.arange(0, duracion_periodo_x, 1 / fs)
    t_y = np.arange(0, duracion_y, 1 / fs)
    t_g = np.arange(0, duracion_g, 1 / fs)
    # -------------------------------------------------------------
    # Definición de señales 
    # -------------------------------------------------------------
    x = (1 + np.cos(2 * np.pi * t_x) / 2) * np.cos(2 * np.pi * 20 * t_x)
    senal_periodo_x = (1 + np.cos(2 * np.pi * t_periodo_x) / 2) * np.cos(2 * np.pi * 20 * t_periodo_x)
    y = np.cos(2 * np.pi * 20 * t_y)
    g = np.cos(2 * np.pi * t_g)

    # Lista de pares (señal, título en formato LaTeX, nombre imagen generada)
    senales = [
        (t_x, x, "$x(t) = (1 + ~ \\frac{1}{2} ~ \\cos(2 \\pi \\cdot t)) ~ \\cos(2 \\pi 20 \\cdot t)$", "5_x.png"),
        (t_periodo_x, senal_periodo_x, "$x(t)~en~un~periodo$", "5_x_en_un_periodo.png"),
        (t_y, y, "$y(t) = \\cos(2 \\pi 20 \\cdot t)$", "5_cos20hz.png"),
        (t_g, g, "$g(t) = \\cos(2 \\pi \\cdot t)$", "5_cos1hz.png"),
    ]

    # -------------------------------------------------------------
    # Generación de un gráfico a la vez
    # -------------------------------------------------------------
    # graficar_senal(senales[0][0], senales[0][1], senales[0][2], senales[0][3], decorar=lambda:resaltar_periodicidad(periodo_x, duracion_x, y_flecha = 1.6), carpeta = "ej5")
    # graficar_senal(senales[1][0], senales[1][1], senales[1][2], senales[1][3], carpeta = "ej5")
    # graficar_senal(senales[2][0], senales[2][1], senales[2][2], senales[2][3], decorar=lambda:resaltar_periodicidad(0.05, duracion_y, y_flecha = 1.1), carpeta = "ej5")
    graficar_senal(senales[3][0], senales[3][1], senales[3][2], senales[3][3], decorar=lambda:resaltar_periodicidad(1, duracion_g, y_flecha = 1.1), carpeta = "ej5")

def main():
    # ejc2()
    # ejc1()
    ejc4()
    # ejc5()

if __name__ == "__main__":
    main()