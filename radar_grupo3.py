#%% Libs and functions
#import os
#os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
from scipy.fft import fft, ifft, fftshift, fftfreq

def fastconv(A,B):
    out_len = len(A)+len(B)-1                     # Longitud de salida
    sizefft = int(2**(np.ceil(np.log2(out_len)))) # Potencia de 2 más cercana
    
    Afilled = np.concatenate((A,np.zeros(sizefft-len(A))))
    Bfilled = np.concatenate((B,np.zeros(sizefft-len(B))))
    
    fftA = fft(Afilled) # Transformada de Fourier de A
    fftB = fft(Bfilled) # Transformada de Fourier de B
    
    fft_out = fftA * fftB # Multiplicación de las transformadas
    out = ifft(fft_out)   # Transformada inversa de Fourier
    
    out = out[0:out_len] # Recorto la salida
    
    return out

## Define constantes físicas y parámetros del radar

c = 3e8              # Velocidad de la luz en m/s
k = 1.380649e-23     # Constante de Boltzmann

fc = 1.3e9          # Frecuencia de portadora 1.3 GHz
fs = 10e6          # Frecuencia de muestreo 10 MHz
Np = 100            # Numero de Intervalos de muestreo
Nint = 10           # Numero de intervalos de integración
NPRIs = Nint*Np     # 
ts = 1/fs           # Periodo de muestreo

Te = 5e-6           # Tiempo de recuperación del Tx 5[μs]
Tp = 10e-6          # Ancho de pulso Tx 10[μs]
BW = 2e6            # Ancho de banda del chirp Tx 2[MHz]
PRF = 1500          # Frecuencia de repetición de pulso 1500[Hz]

wlen = c/fc             # Longitud de onda [m]
kwave = 2*np.pi/wlen    # Numero de onda [rad/m]
PRI = PRF**(-1)         # Periodo de repetición de pulso [s]
ru = (c*(PRI-Tp-Te))/2  # Rango unambiguo [m]
vu_ms = wlen*PRF/2      # Velocidad no ambigua [m/s]
vu_kmh = vu_ms*3.6      # Velocidad no ambigua [km/h]

rank_min = (Tp/2+Te)*c/2    # Rango mínimo [m]
rank_max = 30e3             # Rango maximo [m] (podría ser el Ru)
#rank_max = ru
rank_res = ts*c/2           # Range Step [m]
tmax = 2*rank_max/c         # Tiempo maximo de simulación [s]

# ---------------------------------------------------------------------------------
# Carga la señal de radar desde un archivo CSV y la convierte a un arreglo complejo
# ---------------------------------------------------------------------------------
radar_signal = pd.read_csv('signal_3.csv',index_col=None)
radar_signal = np.array(radar_signal['real']+1j*radar_signal['imag'])
radar_signal = radar_signal.reshape(Np,-1)

# ---------------------------------------------------------------------------------
# Imprime los parámetros calculados
# ---------------------------------------------------------------------------------
print(f'Pulse repetition Interval. PRI = {PRI*1e6:.2f} μs')
print(f'Unambiguous Range. Ru = {ru/1e3:.3f} km')
print(f'Unambiguous Velocity. Vu = {vu_ms:.2f} m/s')
print(f'Unambiguous Velocity. Vu = {vu_kmh:.2f} km/h')
print(f'Minimum Range. Rmin = {rank_min/1e3:.3f} km')
print(f'Maximum Range. Rmin = {rank_max/1e3:.3f} km')


# ---------------------------------------------------------------------------------
# Señales de Tx y Matched Filter
# ---------------------------------------------------------------------------------

# Independant Variables

Npts = int(tmax/ts)                         # Puntos de simulación
t = np.linspace(-tmax/2,tmax/2,Npts)        # Vector de tiempo
ranks = np.linspace(rank_res,rank_max,Npts) # Vector de rangos
f = fftfreq(Npts,ts)                        # Vector de frecuencia

# Señal de transmisión
tx_chirp = np.exp(1j*np.pi*BW/Tp * t**2)    # Tx Linear Chiprs
tx_rect = np.where(np.abs(t)<=Tp/2,1,0)     # Funcion rectangular 
tx_chirp = tx_rect*tx_chirp                 # Chirp rectangular Tx
tx_chirp_f = fft(tx_chirp,norm='ortho')     # Tx Chirp en frecuencia

# Matched Filter
matched_filter = np.conj(np.flip(tx_chirp))
matched_filter_f = fft(matched_filter,norm='ortho')

# Graficos de señales en el tiempo
fig, axes = plt.subplots(2,1,figsize=(8,8),sharex=True)

fig.suptitle('Received Signal')
ax = axes[0]
ax.plot(ranks/1e3,np.real(radar_signal[0]))
ax.plot(ranks/1e3,np.imag(radar_signal[0]))
ax.set_ylabel('Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)

ax = axes[1]
ax.plot(ranks/1e3,np.abs(radar_signal[0]))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)

plt.show()


# ---------------------------------------------------------------------------------
# Convolución de Matched Filter con Señal de Radar. Generación del CHIRP 
# ---------------------------------------------------------------------------------

compressed_signal = []
for t in range(len(radar_signal)):
    compressed_signal_i = fastconv(radar_signal[t],matched_filter)[len(matched_filter)//2:len(matched_filter)*3//2]
    compressed_signal.append(compressed_signal_i)

compressed_signal = np.stack(compressed_signal,axis=0)


# Graficos de señales de radar y comprimida
fig, axes = plt.subplots(2,1,figsize=(8,8))
fig.suptitle('Compressed and Decompressed Signal')

ax = axes[0]
ax.plot(ranks/1000,np.abs(radar_signal[0]))
ax.set_ylabel('Amplitude')
ax.set_xlabel('Rx Raw Signal')
ax.grid(True)

ax = axes[1]
ax.plot(ranks/1000,np.abs(compressed_signal[0]))
ax.set_ylabel('Amplitude')
ax.set_xlabel('Rx Compressed Signal')
ax.grid(True)

plt.show()


#%% ---------------------------------------------------------------------------------
# CFAR Window (Constant False Alarm Rate) - Establece un umbral de detección
# --------------------------------------------------------------------------------- 

# Parametros
gap = 15         
ref = 200        
v_ref = 1       
threshold_factor = 1/(ref*v_ref) # Factor de umbral

# Construcción de la ventana CFAR
cfar1 = np.repeat(threshold_factor,ref/2) 
cfar2 = np.zeros(gap*2) 
cfar3 = np.concatenate((cfar1,cfar2,cfar1)) 

# Grafico de la ventana CFAR
plt.figure(figsize=(10,5))
plt.step(range(len(cfar3)),cfar3)  # Step plot
plt.title('CFAR Window - Absolute Value vs Samples') 
plt.xlabel('Samples') 
plt.ylabel('Absolute Value')
plt.grid(True)
plt.show()


#%% ---------------------------------------------------------------------------------
# MTI SC Filter (Moving Target Indicator de Cancelación Simple)
# ---------------------------------------------------------------------------------

# Calculo de las señales absoluta
abs_radar_signal = np.abs(radar_signal)
#abs_compressed_signal = np.abs(compressed_signal)
gain_MTIsc = 3.85 # Ganancia del filtro MTIsc

# Calculo de la señal MTIsc
MTIsc = (compressed_signal[1])-(compressed_signal[0])
MTIsc_abs = np.abs(MTIsc)
# Calculo del umbral
threshold_MTIsc = gain_MTIsc*fastconv(cfar3,MTIsc_abs)[len(cfar3)//2:len(cfar3)//2 + len(ranks)]

# Proceso de detección 
resta_MTIsc = np.abs(MTIsc_abs) - np.abs(threshold_MTIsc)
sign_MTIsc = np.sign(resta_MTIsc)
target_MTIsc = np.diff(sign_MTIsc)        # 
target_MTIsc = np.append(target_MTIsc,0)    # Agrego un elemento al final para ajustar la longitud de la señal


# Graficos de señales de radar, comprimida, MTIsc 
fig, axes = plt.subplots(4,1,figsize=(6,6), sharex=True)

axes[0].plot(ranks/1000,abs_radar_signal[0],label='Rx t0')
axes[0].plot(ranks/1000,abs_radar_signal[1],label='Rx t1')
axes[0].set_ylabel('Value')
axes[0].set_xlabel('Rx Raw Signal')
axes[0].grid(True) 
axes[0].legend()

axes[1].plot(ranks/1000,np.abs(compressed_signal[0]),label='Compressed t0')
axes[1].plot(ranks/1000,np.abs(compressed_signal[1]),label='Compressed t1')
axes[1].set_ylabel('Value')
axes[1].set_xlabel('Rx Compressed Signal')
axes[1].grid(True) 
axes[1].legend()

axes[2].plot(ranks/1000,np.abs(MTIsc),label='MTIsc')
line2, = axes[2].plot(ranks/1000,np.abs(threshold_MTIsc),label='Threshold')
axes[2].set_ylabel('Value')
axes[2].set_xlabel('MTIsc')
axes[3].grid(True) 
axes[2].legend()

# axes[3].plot(ranks/1000,np.abs(target_MTIsc),label='Target MTIsc')
line3, = axes[3].plot(ranks/1000, np.abs(target_MTIsc))
axes[3].set_ylabel('Value')
axes[3].set_xlabel('Range [km]')
axes[3].grid(True) 
axes[3].legend()


#%% Slider
axcolor = 'lightgoldenrodyellow'
ax_gap = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
slider_gain_MTIsc = Slider(ax_gap, 'MTI SC Gain', 1, 10, valinit=gain_MTIsc)

# Función de actualización para el slider
def update_mti_sc(val):
    gain_MTIsc = slider_gain_MTIsc.val
   
    threshold_MTIsc = gain_MTIsc*fastconv(cfar3,MTIsc_abs)[len(cfar3)//2:len(cfar3)//2 + len(ranks)]

    # Proceso de detección 
    resta_MTIsc = np.abs(MTIsc_abs) - np.abs(threshold_MTIsc)
    sign_MTIsc = np.sign(resta_MTIsc)
    target_MTIsc = np.diff(sign_MTIsc)        # 
    target_MTIsc = np.append(target_MTIsc,0)    # Agrego un elemento al final para ajustar la longitud de la señal


    # Actualizo el gráfico con los cambios
    line2.set_ydata(np.abs(threshold_MTIsc))
    line3.set_ydata(np.abs(target_MTIsc) )   
    
    fig.canvas.draw_idle()
slider_gain_MTIsc.on_changed(update_mti_sc)

plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------------------------------
# MTI DC Filter (Moving Target Indicator de Doble Cancelación)
# Utiliza 3 puntos en el tiempo para realizar la detección
# ---------------------------------------------------------------------------------

gain_MTIdc = 3.85
# Cálculo de la señal MTI de Cancelación Doble
MTIdc = compressed_signal[2]-2*compressed_signal[1]+compressed_signal[0]
MTIdc_abs = np.abs(MTIdc)
# Calcula el umbral usando la ventana CFAR
threshold_MTIdc = gain_MTIdc*fastconv(cfar3,MTIdc_abs)[len(cfar3)//2:len(cfar3)//2 + len(ranks)]

resta_MTIdc = np.abs(MTIdc_abs) - np.abs(threshold_MTIdc)
sign_MTIdc = np.sign(resta_MTIdc)
target_MTIdc = np.diff(sign_MTIdc)  # Derivada de la señal 
target_MTIdc = np.append(target_MTIdc,0)   # Agrego un cero al final para que coincida la longitud con la señal original

# Graficos de señales de radar, comprimida, MTIdc
fig, axes = plt.subplots(4,1,figsize=(6,6), sharex=True)

axes[0].plot(ranks/1000,abs_radar_signal[0],label='Rx t0')
axes[0].plot(ranks/1000,abs_radar_signal[1],label='Rx t1')
axes[0].plot(ranks/1000,abs_radar_signal[2],label='Rx t2')
axes[0].set_ylabel('Value')
axes[0].set_xlabel('Rx Raw Signal')
axes[0].legend()

axes[1].plot(ranks/1000,np.abs(compressed_signal[0]),label='Compressed t0')
axes[1].plot(ranks/1000,np.abs(compressed_signal[1]),label='Compressed t1')
axes[1].plot(ranks/1000,np.abs(compressed_signal[2]),label='Compressed t2')
axes[1].set_ylabel('Value')
axes[1].set_xlabel('Rx Compressed Signal')
axes[1].legend()

axes[2].plot(ranks/1000,np.abs(MTIdc),label='MTIdc')
line2, = axes[2].plot(ranks/1000,np.abs(threshold_MTIdc),label='Threshold MTIdc')
axes[2].set_ylabel('Value')
axes[2].set_xlabel('MTIdc')
axes[2].legend()

#  axes[3].plot(ranks/1000,np.abs(target_MTIdc),label='Target MTIdc')
line3, = axes[3].plot(ranks/1e3, np.abs(target_MTIdc),label='Target MTIdc')
axes[3].set_ylabel('Value')
axes[3].set_xlabel('Range [km]')
axes[3].legend()



#%% Slider
axcolor = 'lightgoldenrodyellow'
ax_gap = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
slider_gain_MTIdc = Slider(ax_gap, 'MTI dc Gain', 1, 10, valinit=gain_MTIdc)

# Función de actualización para el slider
def update_mti_dc(val):
    gain_MTIdc = slider_gain_MTIdc.val
   
    threshold_MTIdc = gain_MTIdc*fastconv(cfar3,MTIdc_abs)[len(cfar3)//2:len(cfar3)//2 + len(ranks)]

    resta_MTIdc = np.abs(MTIdc_abs) - np.abs(threshold_MTIdc)
    sign_MTIdc = np.sign(resta_MTIdc)
    target_MTIdc = np.diff(sign_MTIdc)  # Derivada deL signo
    target_MTIdc = np.append(target_MTIdc,0)   # Agrego un cero al final para que coincida la longitud con la señal original
 
    # Actualizo el gráfico con los cambios
    line2.set_ydata(np.abs(threshold_MTIdc))
    line3.set_ydata(np.abs(target_MTIdc) )   
    
    fig.canvas.draw_idle()

slider_gain_MTIdc.on_changed(update_mti_dc)

plt.tight_layout()
plt.show()

#%% ---------------------------------------------------------------------------------
# STI SC Filter (Stationary Target Indicator de Cancelación Simple)
# ---------------------------------------------------------------------------------

gain_STIsc = 12       # Ganancia del filtro STIsc

# Calculo de la señal STIsc
STIsc = compressed_signal[1]+compressed_signal[0]
STIsc_abs = np.abs(STIsc)

# Calculo del umbral
threshold_STIsc = gain_STIsc*fastconv(cfar3,STIsc_abs)[len(cfar3)//2:len(cfar3)//2 + len(ranks)]

# Proceso de detección
resta_STIsc = np.abs(STIsc_abs) - np.abs(threshold_STIsc)
sign_STIsc = np.sign(resta_STIsc)
target_STIsc = np.diff(sign_STIsc)        
target_STIsc = np.append(target_STIsc,0)    # Agrego un cero al final para que coincida la longitud con la señal original
target_position_STIsc = ranks[target_STIsc == 1]


# Graficos de señales de radar, comprimida, STIsc 
fig, axes = plt.subplots(4,1,figsize=(6,6), sharex=True)

# Señal de radar en dos tiempos diferentes
axes[0].plot(ranks/1000,abs_radar_signal[0],label='Rx t0')
axes[0].plot(ranks/1000,abs_radar_signal[1],label='Rx t1')
axes[0].set_ylabel('Value')
axes[0].set_xlabel('Rx Raw Signal')
axes[0].legend()

# Señal comprimida en dos tiempos diferentes
axes[1].plot(ranks/1000,np.abs(compressed_signal[0]),label='Compressed t0')
axes[1].plot(ranks/1000,np.abs(compressed_signal[1]),label='Compressed t1')
axes[1].set_ylabel('Value')
axes[1].set_xlabel('Rx Compressed Signal')
axes[1].legend()

# Señal STIsc y detección
axes[2].plot(ranks/1000,np.abs(STIsc),label='STIsc')
line2, = axes[2].plot(ranks/1000,np.abs(threshold_STIsc),label='Threshold STIsc')
axes[2].set_ylabel('Value')
axes[2].set_xlabel('STIsc')
axes[2].legend()

# axes[3].plot(ranks/1000,np.abs(target_STIsc),label='Target STIsc')
line3, = axes[3].plot(ranks/1e3, np.abs(target_STIsc))
axes[3].set_ylabel('Value')
axes[3].set_xlabel('Range [km]')
axes[3].legend()



#%% Slider
axcolor = 'lightgoldenrodyellow'
ax_gap = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
slider_gain_STIsc = Slider(ax_gap, 'STI SC Gain', 1, 20, valinit=gain_MTIsc)

# Función de actualización para el slider
def update_sti_sc(val):
    gain_STIsc = slider_gain_STIsc.val
   
    # Calculo del umbral
    threshold_STIsc = gain_STIsc*fastconv(cfar3,STIsc_abs)[len(cfar3)//2:len(cfar3)//2 + len(ranks)]

    # Proceso de detección
    resta_STIsc = np.abs(STIsc_abs) - np.abs(threshold_STIsc)
    sign_STIsc = np.sign(resta_STIsc)
    target_STIsc = np.diff(sign_STIsc)        
    target_STIsc = np.append(target_STIsc,0)  
    # Actualizo el gráfico con los cambios
    line2.set_ydata(np.abs(threshold_STIsc))
    line3.set_ydata(np.abs(target_STIsc) )   
    
    fig.canvas.draw_idle()

slider_gain_STIsc.on_changed(update_sti_sc)

plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------------------------------
# Doppler Processing
# ---------------------------------------------------------------------------------

MTIsc_complete = np.zeros_like(radar_signal)
#print(MTIsc_complete.shape)
for PTR in range (1, Np):
    # Resta de señales comprimidas consecutivas
    MTIsc_complete[PTR] = compressed_signal[PTR] - compressed_signal[PTR-1]

# Transpone la señal para el análisis
MTIsc_transp = MTIsc_complete.T

# Calcula el exponente para la transformada de Fourier
range_sequence = np.arange(1, Np + 1) # crea un array que va desde 1 hasta Np
outer_product = np.outer(range_sequence, range_sequence.T) # crea el producto externo entre el array y su transpuesta
exponent = np.exp(-1j*2*np.pi*outer_product/(Np+1)) 

 
product = (MTIsc_transp @ exponent).T # Multiplica la señal transpuesta por el exponente y luego transpone el resultado
 
velocity = np.linspace(vu_ms/2, -vu_ms/2, Np) # Crea un array de velocidades que va desde vu_ms/2 hasta -vu_ms/2 con Np elementos

# Visualización en 3D del análisis Doppler
X = velocity
Y = ranks/1000
X, Y = np.meshgrid(X, Y)
Z = np.abs(fftshift(product, axes=0)).T

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet')
ax.set_xlabel('Velocity [m/s]')
ax.set_ylabel('Range [km]')
ax.set_zlabel('Amplitude')
ax.set_title('Doppler')
plt.show()

#%%