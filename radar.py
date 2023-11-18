# Trabajo practico radares -Grupo3

#%% Libs and functions
#import os
#os.chdir(os.path.dirname(os.path.abspath('Tp_Grupo3\signal_3.csv')))



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft, fftshift, fftfreq


def fastconv(A,B):
    out_len = len(A)+len(B)-1
    
    # Next nearest power of 2
    sizefft = int(2**(np.ceil(np.log2(out_len))))
    
    Afilled = np.concatenate((A,np.zeros(sizefft-len(A))))
    Bfilled = np.concatenate((B,np.zeros(sizefft-len(B))))
    
    fftA = fft(Afilled)
    fftB = fft(Bfilled)
    
    fft_out = fftA * fftB
    out = ifft(fft_out)
    
    out = out[0:out_len]
    
    return out

#%% Parameters

c = 3e8 # speed of light [m/s]
k = 1.380649e-23 # Boltzmann

fc = 1.3e9 # Carrier freq
fs = 10e6 # Sampling freq
Np = 100 # Intervalos de sampling
Nint = 10
NPRIs = Nint*Np
ts = 1/fs

Te = 5e-6 # Tx recovery Time[s]
Tp = 10e-6 # Tx Pulse Width [s]
BW = 2e6 # Tx Chirp bandwidth [Hz]
PRF = 1500 # Pulse repetition Frequency [Hz]

wlen = c/fc # Wavelength [m]
kwave = 2*np.pi/wlen # Wavenumber [rad/m]
PRI = PRF**(-1) # Pulse repetition interval [s]
ru = (c*(PRI-Tp-Te))/2 # Unambigous Range [m]
vu_ms = wlen*PRF/2 # Unambigous Velocity [m/s]
vu_kmh = vu_ms*3.6 # Unambigous Velocity [km/h]

rank_min = (Tp/2+Te)*c/2 # Minimum Range [m]
rank_max = 30e3 # Maximum Range [m] (podría ser el Ru)
#rank_max = ru
rank_res = ts*c/2 # Range Step [m]
tmax = 2*rank_max/c # Maximum Simulation Time



radar_signal = pd.read_csv('signal_3.csv',index_col=None)
radar_signal = np.array(radar_signal['real']+1j*radar_signal['imag'])
radar_signal = radar_signal.reshape(Np,-1)

print(f'Pulse repetition Interval. PRI = {PRI*1e6:.2f} μs')
print(f'Unambiguous Range. Ru = {ru/1e3:.3f} km')
print(f'Unambiguous Velocity. Vu = {vu_ms:.2f} m/s')
print(f'Unambiguous Velocity. Vu = {vu_kmh:.2f} km/h')
print(f'Minimum Range. Rmin = {rank_min/1e3:.3f} km')
print(f'Maximum Range. Rmin = {rank_max/1e3:.3f} km')

#%% Signals

# Independant Variables

Npts = int(tmax/ts) # Simulation Points
t = np.linspace(-tmax/2,tmax/2,Npts)
ranks = np.linspace(rank_res,rank_max,Npts) # Range Vector
f = fftfreq(Npts,ts) # Freq Vector

# Tx Signal

tx_chirp = np.exp(1j*np.pi*BW/Tp * t**2) # Tx Linear Chiprs (t)
tx_rect = np.where(np.abs(t)<=Tp/2,1,0) # Rect Function
tx_chirp = tx_rect*tx_chirp # Tx Chirp Rectangular
tx_chirp_f = fft(tx_chirp,norm='ortho') # Tx Chirp (f)

# Matched Filter

matched_filter = np.conj(np.flip(tx_chirp))
matched_filter_f = fft(matched_filter,norm='ortho')



#%% Plot Signals

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
#%% Convolución de Matched Filter con txsignal
 
compressed_signal = []
for t in range(len(radar_signal)):
    compressed_signal_i = fastconv(radar_signal[t], matched_filter)
    # Recortar la señal convolucionada
    start_idx = int(len(matched_filter)/2)
    end_idx = start_idx + len(radar_signal[t])
    compressed_signal_i = compressed_signal_i[start_idx:end_idx]
    compressed_signal.append(compressed_signal_i)

compressed_signal = np.stack(compressed_signal, axis=0)
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
fig.suptitle('Convoluted Signal')

# Graficar la parte real
ax = axes[0]
ax.plot(ranks / 1e3, compressed_signal[0].real, label='Real Part')
ax.set_ylabel('Amplitude')
ax.grid(True)
ax.legend()

# Graficar la parte imaginaria
ax = axes[1]
ax.plot(ranks / 1e3, compressed_signal[0].imag, label='Imaginary Part')
ax.set_ylabel('Amplitude')
ax.grid(True)
ax.legend()

# Graficar la magnitud
ax = axes[2]
ax.plot(ranks / 1e3, np.abs(compressed_signal[0]), label='Magnitude')
ax.set_ylabel('Magnitude')
ax.set_xlabel('Range [km]')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()


#%% CFAR Window
# Se implementa un algoritmo de CFAR (Constant False Alarm Rate)
# De esta manera se establece un umbral de detección para la señal comprimida


# Parametros
gap = 15 # Gap entre las celdas de referencia y las celdas de protección
ref = 200
v_ref = 1
threshold_factor = 1/(ref*v_ref) # Factor de umbral


cfar1 = np.repeat(threshold_factor,ref/2)
cfar2 = np.zeros(gap*2)
cfar3 = np.concatenate((cfar1,cfar2,cfar1))


plt.figure(figsize=(10,5))
# plt.plot(cfar3)
plt.step(range(len(cfar3)),cfar3)  # Step plot
plt.title('CFAR Window - Absolute Value vs Samples') #
plt.xlabel('Samples') #
plt.ylabel('Absolute Value')
plt.grid(True)
plt.show()




#%% MTI SC Filter
# Se aplica un cancelador simple de movimiento (MTI) sobre la señal comprimida

abs_radar_signal = np.abs(radar_signal)
abs_compressed_signal = np.abs(compressed_signal)
gain_MTIsc = 3.85
MTIsc = (abs_compressed_signal[1])-(abs_compressed_signal[0])
MTIsc_abs = np.abs(MTIsc)
threshold_MTIsc = gain_MTIsc*fastconv(cfar3,MTIsc_abs)[len(cfar3)//2:len(cfar3)//2 + len(ranks)]


resta_MTIsc = np.abs(MTIsc_abs) - np.abs(threshold_MTIsc)
sign_MTIsc = np.sign(resta_MTIsc)
diff_sign_MTIsc = np.diff(sign_MTIsc) # Derivada de la señal
diff_MTIsc = np.diff(sign_MTIsc) # Segunda derivada de la señal
diff_MTIsc = np.append(diff_MTIsc,0) # Agrego un cero al final para que coincida la longitud con la señal original


# Plot del MTI
fig, axes = plt.subplots(4,1,figsize=(10,20), sharex=True)


axes[0].plot(ranks/1000,abs_radar_signal[0],label='Rx t0')
axes[0].plot(ranks/1000,abs_radar_signal[1],label='Rx t1')
axes[0].set_ylabel('Value')
axes[0].set_xlabel('Rx Raw Signal')
axes[0].legend()


axes[1].plot(ranks/1000,np.abs(compressed_signal[0]),label='Compressed t0')
axes[1].plot(ranks/1000,np.abs(compressed_signal[1]),label='Compressed t1')
axes[1].set_ylabel('Value')
axes[1].set_xlabel('Rx Compressed Signal')
axes[1].legend()


axes[2].plot(ranks/1000,np.abs(MTIsc),label='MTIsc')
axes[2].plot(ranks/1000,np.abs(threshold_MTIsc),label='Threshold')
axes[2].set_ylabel('Value')
axes[2].set_xlabel('MTIsc')
axes[2].legend()


axes[3].plot(ranks/1000,np.abs(diff_MTIsc),label='Diff MTIsc')
axes[3].set_ylabel('Value')
axes[3].set_xlabel('Range [km]')
axes[3].legend()


plt.tight_layout()
plt.show()


#%% STI SC Filter


gain_STIsc = 3.85
STIsc = (compressed_signal[1])+(compressed_signal[0])
STIsc_abs = np.abs(STIsc)
threshold_STIsc = gain_STIsc*fastconv(cfar3,STIsc_abs)[len(cfar3)//2:len(cfar3)//2 + len(ranks)]


resta_STIsc = np.abs(STIsc_abs) - np.abs(threshold_STIsc)
sign_STIsc = np.sign(resta_STIsc)
diff_sign_STIsc = np.diff(sign_STIsc) # Derivada de la señal
diff_STIsc = np.diff(sign_STIsc) # Segunda derivada de la señal
diff_STIsc = np.append(diff_STIsc,0) # Agrego un cero al final para que coincida la longitud con la señal original


# Plot del STI
fig, axes = plt.subplots(4,1,figsize=(10,20), sharex=True)


axes[0].plot(ranks/1000,abs_radar_signal[0],label='Rx t0')
axes[0].plot(ranks/1000,abs_radar_signal[1],label='Rx t1')
axes[0].set_ylabel('Value')
axes[0].set_xlabel('Rx Raw Signal')
axes[0].legend()


axes[1].plot(ranks/1000,np.abs(compressed_signal[0]),label='Compressed t0')
axes[1].plot(ranks/1000,np.abs(compressed_signal[1]),label='Compressed t1')
axes[1].set_ylabel('Value')
axes[1].set_xlabel('Rx Compressed Signal')
axes[1].legend()


axes[2].plot(ranks/1000,np.abs(STIsc),label='STIsc')
axes[2].plot(ranks/1000,np.abs(threshold_STIsc),label='Threshold STIsc')
axes[2].set_ylabel('Value')
axes[2].set_xlabel('STIsc')
axes[2].legend()


axes[3].plot(ranks/1000,np.abs(diff_STIsc),label='Diff STIsc')
axes[3].set_ylabel('Value')
axes[3].set_xlabel('Range [km]')
axes[3].legend()


plt.tight_layout()
plt.show()
# %% Doppler 
# Cálculo de la frecuencia Doppler
doppler_freq = fftfreq(Npts, ts)
tiempo_de_retardo = 0.0  
umbral_doppler = 100.0

MTIsc_complete = np.zeros_like(radar_signal)


mti_sc_doppler=threshold_MTIsc*np.exp(1j*2*np.pi*doppler_freq*tiempo_de_retardo)

#for t in range (gap, len(ranks)-gap):
 #  MTIsc_complete[:,t] = compressed_signal[:,t] - compressed_signal[:,t-1]

MTIsc_transp = MTIsc_complete.T

exponent = np.exp(-1j*2*np.pi*np.outer(np.arange(1, Np+1), np. arange(1, Np+1).T)/(Np+1))
product = (MTIsc_transp @ exponent).T

doppler_filter = np.abs(doppler_freq)<umbral_doppler
MTI_sc_doppler_filtered=product*doppler_filter


velocity = np.linspace(vu_ms/2, -vu_ms/2, Np)
rango_mesh, velocidad_mesh = np.meshgrid(ranks, velocity)

#Plot doppler

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(rango_mesh, velocidad_mesh, np.abs(MTI_sc_doppler_filtered), cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=False)

# Etiquetas y título
ax.set_xlabel('Velocidad [m/s]')
ax.set_ylabel('Rango [km]')
ax.set_zlabel('Producto Doppler')
ax.set_title('Detección de blancos con Producto Doppler')

# Barra de colores
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)

# Mostrar el gráfico
plt.show()


#%%