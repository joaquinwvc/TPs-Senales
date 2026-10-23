import matplotlib.pyplot as plt
import numpy as np

def graficar_espectro_simetrico(frecuencias, amplitudes, titulo='Espectro Analítico (Vista dividida)', margen=100, nombre_archivo=None):
    """
    Grafica un espectro simétrico omitiendo las frecuencias centrales vacías y opcionalmente guarda la imagen.
    """
    frecuencias = np.array(frecuencias)
    amplitudes = np.array(amplitudes)
    
    # Crear figura con dos paneles que comparten el eje Y
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    
    # Graficar los datos en ambos
    ax1.stem(frecuencias, amplitudes, basefmt=" ")
    ax2.stem(frecuencias, amplitudes, basefmt=" ")
    
    # Filtrar frecuencias positivas y negativas
    freqs_negativas = frecuencias[frecuencias < 0]
    freqs_positivas = frecuencias[frecuencias > 0]
    
    # Configurar panel izquierdo (negativas)
    if len(freqs_negativas) > 0:
        ax1.set_xlim(np.min(freqs_negativas) - margen, np.max(freqs_negativas) + margen)
        ax1.set_xticks(np.unique(freqs_negativas))
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylabel('Amplitud')
    
    # Configurar panel derecho (positivas)
    if len(freqs_positivas) > 0:
        ax2.set_xlim(np.min(freqs_positivas) - margen, np.max(freqs_positivas) + margen)
        ax2.set_xticks(np.unique(freqs_positivas))
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Textos generales
    fig.suptitle(titulo)
    fig.text(0.5, 0.01, 'Frecuencia (Hz)', ha='center')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar la imagen si se proporciona un nombre de archivo
    if nombre_archivo:
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        
    plt.show()

# --- Ejemplo de uso ---
frec_hz = [-8200, -8000, -7800, 7800, 8000, 8200]
amp = [np.pi/4, np.pi, np.pi/4, np.pi/4, np.pi, np.pi/4]

# Grafica y guarda la imagen como "espectro_analitico.png"
graficar_espectro_simetrico(frec_hz, amp, titulo=r'Espectro $S_0(f)$', nombre_archivo='1_espectro_So(w).png')