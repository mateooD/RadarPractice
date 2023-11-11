import numpy as np
import matplotlib.pyplot as plt

# Frecuencia en kHz
f_portadora_kHz = 1010 *10**3  # Frecuencia de la portadora en kHz
f_modulante = 10*10**3  # Frecuencia de la señal modulante en Hz
m = 1  # Índice de modulación

# Convertir la frecuencia de portadora a Hz
#f_portadora = f_portadora_kHz * 1000.0

# Tiempo
t = np.linspace(0, 1, 10000)  # Genera un vector de tiempo de 1 segundo con 1000 puntos

# Señal de portadora y señal modulante
portadora = np.cos(2 * np.pi * f_portadora_kHz * t)
modulante = np.cos(2 * np.pi * f_modulante * t)

# Señal modulada AM
senal_am = (1 + m * modulante) * portadora

# Cálculo de las bandas laterales centradas en la frecuencia de la portadora
banda_superior = m * np.cos(2 * np.pi * f_modulante * t)
banda_inferior = -m * np.cos(2 * np.pi * f_modulante * t)

# Transformada de Fourier de las bandas laterales
fft_banda_superior = np.fft.fft(banda_superior)
fft_banda_inferior = np.fft.fft(banda_inferior)
frecuencias = np.fft.fftfreq(len(t), t[1] - t[0])

# Desplazar las frecuencias alrededor de la frecuencia de la portadora
frecuencias_desplazadas = np.fft.fftshift(frecuencias)
fft_banda_superior_desplazada = np.fft.fftshift(fft_banda_superior)
fft_banda_inferior_desplazada = np.fft.fftshift(fft_banda_inferior)

# Gráfica de la señal AM, la portadora y las bandas laterales en el dominio del tiempo
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.title("Señal Modulante")
plt.plot(t, modulante)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")

plt.subplot(4, 1, 2)
plt.title("Portadora")
plt.plot(t, portadora)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")

plt.subplot(4, 1, 3)
plt.title("Señal Modulada AM")
plt.plot(t, senal_am)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")

# Gráfica de las bandas laterales en el dominio de la frecuencia
plt.subplot(4, 1, 4)
plt.title("Bandas Laterales - Dominio de la Frecuencia")
plt.plot(frecuencias_desplazadas, np.abs(fft_banda_superior_desplazada), label="Banda Superior")
plt.plot(frecuencias_desplazadas, np.abs(fft_banda_inferior_desplazada), label="Banda Inferior")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.legend()

plt.tight_layout()
plt.show()