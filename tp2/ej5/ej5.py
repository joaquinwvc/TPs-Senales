import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import warnings
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt

# -------------------------------------------------------------
# Configuración de parámetros y Funciones Base (de ej4_2.py)
# -------------------------------------------------------------
def leer_audio(ruta_archivo):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        fs, data = wavfile.read(ruta_archivo)
    return fs, data

def espectrograma(m, fs, ventana_ms, nfft=None, overlap=0.5):
    nperseg = int(round(ventana_ms * 1e-3 * fs))
    if nfft is None:
        nfft = nperseg
    paso = max(1, nperseg - int(nperseg * overlap))

    w = np.hanning(nperseg)                                 
    segmentos = sliding_window_view(m, nperseg)[::paso]     
    X = np.fft.fft(segmentos * w, n=nfft, axis=1)           
    X = X[:, : nfft // 2 + 1]                               

    S = np.abs(X) ** 2 / (fs * np.sum(w ** 2))              
    S[:, 1:-1] *= 2                                         

    f = np.abs(np.fft.fftfreq(nfft, d=1 / fs)[: nfft // 2 + 1])
    t = (np.arange(len(segmentos)) * paso + nperseg / 2) / fs   
    return f, t, S.T

def graficar_espectrograma(f, t, S, titulo, ruta, fmin=0, fmax=8000, t_min=None, t_max=None):
    S_db = 10 * np.log10(S + 1e-12)
    vmax = S_db.max()
    vmin = vmax - 80

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, S_db, shading="gouraud", vmin=vmin, vmax=vmax)
    plt.xlim(t_min, t_max)      
    plt.ylim(fmin, fmax)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.title(titulo)
    plt.colorbar(label="Potencia [dB]")
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    # plt.show()

def filtro_pasabajos(v, fs, f_corte=8000, orden=8):
    sos = butter(orden, f_corte, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, v)

# ------------------------------------------------------------- Ejercicio 5

def recuperar_emisora_desconocida(w, fs, fc, fase, f_corte=4000):
    """Demodula sincronizando la fase y elimina la continua."""
    t = np.arange(len(w)) / fs
    
    # 1) Multiplicación por portadora local con la fase corregida
    v = w * np.cos(2 * np.pi * fc * t + fase)
    
    # 2) Filtro pasa-bajos para eliminar la réplica en alta frecuencia
    y = filtro_pasabajos(v, fs, f_corte)
    
    # 3) Quitar la continua (el valor medio)
    y = y - np.mean(y)
    
    # 4) Normalizar para máxima amplitud sin saturar
    return y / np.max(np.abs(y))

def ej5_1_espectrograma(w, fs):
    os.makedirs("1", exist_ok=True)
    ventana = 100
    nfft = 16384
    f, t, S = espectrograma(w, fs, ventana, nfft=nfft)
    # graficar_espectrograma(f, t, S,
    #                        f"Espectrograma de emisoras.wav (alta resolucion frecuencial) - ventana de {ventana} ms",
    #                        "1/espectrograma_emisoras_hrf.png", fmax=47000, t_min=0, t_max=len(w)/fs)

    ventana = 10
    nfft= 2048
    f, t, S = espectrograma(w, fs, ventana, nfft=nfft, overlap = 0.5)
    graficar_espectrograma(f, t, S,
                           f"Espectrograma de emisoras.wav (alta resolucion temporal) - ventana de {ventana} ms",
                           "1/espectrograma_emisoras_hrt.png", fmax=47000, t_min=0, t_max=len(w)/fs)


def ej5_2_3_demodular_todas(w, fs, emisoras_info):
    os.makedirs("3", exist_ok=True)
    
    for i, info in enumerate(emisoras_info):
        fc_aprox = info['frecuencia']
        f_corte_base = info['ancho_banda'] / 2  # El filtro pasabajos usa la mitad del ancho de banda AM
        t_start = info['t_start']
        t_end = info['t_end']
        
        # 1. Aislar el segmento donde la emisora transmite a plena potencia
        idx_start = int(t_start * fs)
        idx_end = int(t_end * fs)
        w_activa = w[idx_start:idx_end]
        
        # 2. Calcular FFT solo sobre el segmento activo
        N_act = len(w_activa)
        X_act = np.fft.fft(w_activa)
        freqs_act = np.fft.fftfreq(N_act, 1/fs)
        
        f_pos = freqs_act[:N_act//2]
        X_pos = X_act[:N_act//2]
        
        # 3. Encontrar el índice de la frecuencia y buscar el pico exacto (+/- 50 Hz)
        idx_cercano = np.argmin(np.abs(f_pos - fc_aprox))
        rango = int(50 / (fs / N_act))
        idx_rango = slice(max(0, idx_cercano - rango), min(len(f_pos), idx_cercano + rango))
        idx_max_local = idx_rango.start + np.argmax(np.abs(X_pos[idx_rango]))
        
        fc_real = f_pos[idx_max_local]
        
        # La fase extraída corresponde al instante t_start
        fase_segmento = np.angle(X_pos[idx_max_local])
        
        # 4. Rebobinar matemáticamente la fase al instante t=0
        fase_real = fase_segmento - (2 * np.pi * fc_real * t_start)
        
        print(f"Emisora {i+1}: f_c = {fc_real:.2f} Hz | Fase compensada = {np.degrees(fase_real):.2f}° | Filtro = {f_corte_base} Hz")
        
        # 5. Demodular la señal completa usando la fase corregida al origen
        m_hat = recuperar_emisora_desconocida(w, fs, fc_real, fase_real, f_corte=f_corte_base)
        
        # 6. Graficar el espectrograma del audio recuperado
        ventana = 40
        nfft = 4096
        f, t, S = espectrograma(m_hat, fs, ventana, nfft=nfft)
        graficar_espectrograma(
            f, t, S,
            f"Espectrograma de emisora {i+1} ({int(fc_real)} Hz) - ventana de {ventana} ms",
            f"3/espectrograma_emisora_{i+1}.png",
            fmax=10500,
            t_min=0,
            t_max=len(w)/fs
        )
        
        # 7. Guardar audio
        ruta_salida = f"3/emisora_{i+1}_f{int(fc_real)}Hz.wav"
        wavfile.write(ruta_salida, fs, (0.9 * m_hat).astype(np.float32))
        print(f" -> Guardado: {ruta_salida}")

def main():
    archivo = "emisoras.wav"
    try:
        fs, w = leer_audio(archivo)
    except FileNotFoundError:
        print(f"No se encontró '{archivo}'. Asegúrate de que esté en la carpeta.")
        return
        
    print(f"Archivo {archivo} cargado (fs = {fs} Hz).")
    
    # 1. Generar espectrogramas (comentados o no según necesites)
    ej5_1_espectrograma(w, fs)
    
    # 2. Configurar frecuencias visualizadas y sus anchos de banda totales
    emisoras_info = [
        {'frecuencia': 9000, 'ancho_banda': 10000, 't_start': 1, 't_end': 20},
        {'frecuencia': 17000, 'ancho_banda': 7000, 't_start': 1, 't_end': 15},
        {'frecuencia': 26000, 'ancho_banda': 10000, 't_start': 5, 't_end': 25},
        {'frecuencia': 38000, 'ancho_banda': 10000, 't_start': 10, 't_end': 18}
    ]
    
    # 3. Extraer fase, demodular y guardar cada señal
    # ej5_2_3_demodular_todas(w, fs, emisoras_info)

if __name__ == "__main__":
    main()
