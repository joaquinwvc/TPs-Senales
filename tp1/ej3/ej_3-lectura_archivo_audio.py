import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def analizar_y_graficar_audio(ruta_archivo):
    # 1. Leer el archivo
    # fs es la frecuencia de muestreo (sample rate)
    # data contiene las muestras del audio
    fs, data = wavfile.read(ruta_archivo)

    # 2. Obtener la frecuencia de muestreo
    print(f"--- Análisis de: {ruta_archivo} ---")
    print(f"2. Frecuencia de muestreo: {fs} Hz")

    # Determinar si es Mono o Estéreo
    if len(data.shape) == 1:
        canales = 1
        print("El audio es Mono (1 canal).")
    else:
        canales = data.shape[1]
        print(f"El audio es Estéreo ({canales} canales).")

    # 3. Obtener la cantidad total de muestras
    muestras_totales = data.shape[0]
    print(f"3. Cantidad total de muestras: {muestras_totales}")

    # 4. Calcular la duración del audio (Duración = Muestras / Frecuencia de muestreo)
    duracion = muestras_totales / fs
    print(f"4. Duración del audio: {duracion:.3f} segundos")

    # Crear el vector de tiempo en segundos para el eje X
    tiempo = np.linspace(0., duracion, muestras_totales)

    # Configurar los gráficos (creamos una figura con 3 subgráficos)
    # Si es estéreo, graficaremos solo el canal izquierdo (canal 0) para simplificar la vista, 
    # pero mostraremos ambos en el primer gráfico si lo deseas.
    
    # Para manejar estéreo en los gráficos de forma sencilla:
    if canales == 2:
        # Extraemos el canal 1 (izquierdo) y el canal 2 (derecho)
        canal_izq = data[:, 0]
        canal_der = data[:, 1]
        señal_a_graficar = canal_izq # Usaremos el izquierdo para los recortes
        titulo_extra = "(Canal Izquierdo)"
    else:
        señal_a_graficar = data
        titulo_extra = ""

    plt.figure(figsize=(12, 10))

    # --- 5. Graficar la señal completa ---
    plt.subplot(3, 1, 1)
    if canales == 2:
        plt.plot(tiempo, canal_izq, label="Canal Izquierdo", color='blue', alpha=0.7)
        plt.plot(tiempo, canal_der, label="Canal Derecho", color='orange', alpha=0.7)
        plt.legend()
    else:
        plt.plot(tiempo, señal_a_graficar, color='blue')
    plt.title("Señal de audio completa")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid(True)

    # --- 6. Graficar únicamente los primeros 50 ms (0.05 segundos) ---
    muestras_50ms = int(0.05 * fs)
    # Verificamos que el audio dure al menos 50ms
    if muestras_totales >= muestras_50ms:
        plt.subplot(3, 1, 2)
        plt.plot(tiempo[:muestras_50ms], señal_a_graficar[:muestras_50ms], color='green')
        plt.title(f"Primeros 50 ms {titulo_extra}")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Amplitud")
        plt.grid(True)

    # --- 7. Graficar únicamente los primeros 500 ms (0.5 segundos) ---
    muestras_500ms = int(0.5 * fs)
    # Verificamos que el audio dure al menos 500ms
    if muestras_totales >= muestras_500ms:
        plt.subplot(3, 1, 3)
        plt.plot(tiempo[:muestras_500ms], señal_a_graficar[:muestras_500ms], color='red')
        plt.title(f"Primeros 500 ms {titulo_extra}")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Amplitud")
        plt.grid(True)

    plt.tight_layout(h_pad=2.0)
    plt.show()

if __name__ == "__main__":
    # ¡CAMBIA ESTE NOMBRE POR EL DE TU ARCHIVO DESCARGADO!
    nombre_archivo = "Windows_XP_Startup.wav" 
    
    try:
        analizar_y_graficar_audio(nombre_archivo)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{nombre_archivo}'. Asegúrate de que esté en la misma carpeta que este script.")
