import numpy as np
import matplotlib.pyplot as plt

def graficar_espectro_fft(senal, fs, xmin, xmax, n_fft=None, titulo="Espectro Numérico", nombre_archivo=None):
    """
    Calcula la FFT de una señal especificando la cantidad de puntos N (Zero-Padding si N > len(senal)),
    centra las frecuencias y grafica un sector del espectro.
    """
    # Si no se especifica n_fft, se usa la longitud original de la señal
    if n_fft is None:
        N = len(senal)
    else:
        N = n_fft
    
    # Calcular la FFT y normalizar su amplitud
    fft_calculada = np.fft.fft(senal, n=N)
    modulo = np.abs(fft_calculada) / len(senal)
    
    # Generar el eje de frecuencias y centrar el espectro
    frecuencias = np.fft.fftfreq(N, d=1/fs)
    frecuencias_centradas = np.fft.fftshift(frecuencias)
    modulo_centrado = np.fft.fftshift(modulo)
    
    # Crear la figura
    plt.figure(figsize=(10, 4))
    plt.plot(frecuencias_centradas, modulo_centrado, color='blue')
    
    # Configurar los límites y etiquetas
    plt.xlim(xmin, xmax)
    plt.title(titulo)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud Normalizada')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Guardar la imagen si se provee un nombre de archivo
    if nombre_archivo:
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        
    plt.show()

#ejercicio 1.2
fs = 96000
t = np.arange(0, 2, 1/fs)
s0 = (1 + 0.5 * np.cos(2 * np.pi * 200 * t)) * np.cos(2 * np.pi * 8000 * t)

# Llamada a la función definiendo el zoom entre 7700 y 8300 Hz
# graficar_espectro_fft(senal=s0, 
#                       fs=fs, 
#                       xmin=4000, 
#                       xmax=12000, 
#                       titulo=r'Espectro Numérico de $s_0(t)$', 
#                       nombre_archivo='1_espectro_So(w)_fft.png')

#ejercicio 1.3
t_2m = np.arange(0, 0.002, 1/fs)
s0_2m = (1 + 0.5 * np.cos(2 * np.pi * 200 * t_2m)) * np.cos(2 * np.pi * 8000 * t_2m)
# graficar_espectro_fft(senal=s0_2m, 
#                       fs=fs, 
#                       xmin=4000, 
#                       xmax=12000, 
#                       titulo=r'Espectro Numérico de $s_0(t)$ de duracion $2ms$', 
#                       nombre_archivo='1_espectro_So(w)_fft_2m.png')

# t_5m = np.arange(0, 0.005, 1/fs)
# s0_5m = (1 + 0.5 * np.cos(2 * np.pi * 200 * t_5m)) * np.cos(2 * np.pi * 8000 * t_5m)
# graficar_espectro_fft(senal=s0_5m, 
#                       fs=fs, 
#                       xmin=4000, 
#                       xmax=12000, 
#                       titulo=r'Espectro Numérico de $s_0(t)$ de duracion $5ms$', 
#                       nombre_archivo='1_espectro_So(w)_fft_5m.png')

# t_10m = np.arange(0, 0.010, 1/fs)
# s0_10m = (1 + 0.5 * np.cos(2 * np.pi * 200 * t_10m)) * np.cos(2 * np.pi * 8000 * t_10m)
# graficar_espectro_fft(senal=s0_10m, 
#                       fs=fs, 
#                       xmin=4000, 
#                       xmax=12000, 
#                       titulo=r'Espectro Numérico de $s_0(t)$ de duracion $10ms$', 
#                       nombre_archivo='1_espectro_So(w)_fft_10m.png')

# t_50m = np.arange(0, 0.050, 1/fs)
# s0_50m = (1 + 0.5 * np.cos(2 * np.pi * 200 * t_50m)) * np.cos(2 * np.pi * 8000 * t_50m)
# graficar_espectro_fft(senal=s0_50m, 
#                       fs=fs, 
#                       xmin=4000, 
#                       xmax=12000, 
#                       titulo=r'Espectro Numérico de $s_0(t)$ de duracion $50ms$', 
#                       nombre_archivo='1_espectro_So(w)_fft_50m.png')

#ejercicio 1.4
# t_2m = np.arange(0, 0.002, 1/fs)
# s0_2m = (1 + 0.5 * np.cos(2 * np.pi * 200 * t_2m)) * np.cos(2 * np.pi * 8000 * t_2m)

#n_nftt = 192
# graficar_espectro_fft(senal=s0_2m, 
#                       fs=fs, 
#                       xmin=4000, 
#                       xmax=12000, 
#                       n_fft= 192,
#                       titulo=r'Espectro Numérico de $s_0(t)$ de duracion $2ms$ - 192 puntos', 
#                       nombre_archivo='1_espectro_So(w)_fft_2m_192pts.png')
#n_nftt = 1024
# graficar_espectro_fft(senal=s0_2m, 
#                       fs=fs, 
#                       xmin=4000, 
#                       xmax=12000, 
#                       n_fft= 1024,
#                       titulo=r'Espectro Numérico de $s_0(t)$ de duracion $2ms$ - 1024 puntos', 
#                       nombre_archivo='1_espectro_So(w)_fft_2m_1024pts.png')
#n_nftt = 8192
# graficar_espectro_fft(senal=s0_2m, 
#                       fs=fs, 
#                       xmin=4000, 
#                       xmax=12000, 
#                       n_fft= 8192,
#                       titulo=r'Espectro Numérico de $s_0(t)$ de duracion $2ms$ - 8192 puntos', 
#                       nombre_archivo='1_espectro_So(w)_fft_2m_8192pts.png')

mu1 = 0.3
mu2 = 1
s0_mu1 = (1 + mu1 * np.cos(2 * np.pi * 200 * t)) * np.cos(2 * np.pi * 8000 * t)
graficar_espectro_fft(senal=s0_mu1, 
                      fs=fs, 
                      xmin=7500, 
                      xmax=8500, 
                      titulo=r'Espectro Numérico de $s_0(t)$ con $\mu = 0,3$', 
                      nombre_archivo='1_espectro_So(w)_fft_mu1.png')

s0_mu2 = (1 + mu2 * np.cos(2 * np.pi * 200 * t)) * np.cos(2 * np.pi * 8000 * t)
graficar_espectro_fft(senal=s0_mu2, 
                      fs=fs, 
                      xmin=7500, 
                      xmax=8500, 
                      titulo=r'Espectro Numérico de $s_0(t)$ con $\mu = 1$', 
                      nombre_archivo='1_espectro_So(w)_fft_mu2.png')

s1 = np.cos(2 * np.pi * 200 * t) * np.cos(2 * np.pi * 8000 * t)
graficar_espectro_fft(senal=s1, 
                      fs=fs, 
                      xmin=7500, 
                      xmax=8500, 
                      titulo=r'Espectro Numérico de $s_1(t)$', 
                      nombre_archivo='1_espectro_S1(w)_fft.png')