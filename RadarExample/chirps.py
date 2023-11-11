# -*- coding: utf-8 -*-
"""

@author: fdadam
"""
#%% Libs and functions
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import spectrogram
from matplotlib.gridspec import GridSpec

c = 3e8 # speed of light [m/s]
fc = 1.3e9 # Carrier freq
fs = 100e6 # Sampling freq
ts = 1/fs # Sampling Time

Tp = 10e-6 # Pulse Width
f0 = 0e6 # Chirp Initial Freq
BW = 20e6 # Chirp bandwidth
PRF = 1500 # Pulse repetition Frequency [Hz]

wlen = c/fc # Wavelength
kwave = 2*np.pi/wlen
PRI = PRF**(-1) # Pulse repetition interval [s]
ru = (c*(PRI-Tp))/2 # Unambigous Range
vu = wlen*PRF/2 # Unambigous Velocity


#%% Example signals

Npts = int(Tp/ts)
t = np.linspace(-Tp/2,Tp/2,Npts)

# Acomodar array de freqs para plot
f = fftfreq(Npts,ts)
f = np.concatenate((f[Npts//2::],f[:Npts//2]))# Acomodar para plot

# Generar signals

chirp_ref = np.exp(1j*2*np.pi*(f0*t+BW/(2*Tp) * t**2))
matched_filter = np.conj(np.flip(chirp_ref))
#matched_filter = np.flip((chirp_ref))
#matched_filter = np.exp(-1j*np.pi*BW/Tp * t**2)

chirp_ref_f = fftshift(fft(chirp_ref))
matched_filter_f = fftshift(fft(matched_filter))

# Para que la SINC resultante de la convolución quede centrada en 0
# la FFT debe tener una cantidad de puntos que sea un potencia de 2
# La cantidad de puntos de la convolcion depende de las señales de entrada

# Hacer las FFT full resolution
sizefft = len(chirp_ref_f)+len(matched_filter_f)-1
sizefft = int(2**(np.ceil(np.log2(sizefft))))
chirp_ref_f_full = (fft(chirp_ref,n=sizefft))
matched_filter_f_full = (fft(matched_filter,n=sizefft))

f_full = fftfreq(sizefft,ts)
f_full = np.concatenate((f_full[sizefft//2::],f_full[0:sizefft//2]))# Acomodar para plot

# Convolucionar
chirp_comp_f = fftshift(chirp_ref_f_full*matched_filter_f_full)
chirp_comp = ifft((chirp_comp_f))


# Quitar colas de convolucion
chirp_comp = chirp_comp[len(matched_filter)//2:len(chirp_ref)+len(matched_filter)//2]


# Plot chirp ref centered in 0

fig = plt.figure(layout="tight",figsize=(16,9))
gs = GridSpec(3, 3, figure=fig)

ax = fig.add_subplot(gs[0,0])
ax.plot(t*1e6,chirp_ref.real,label=r'$\Re$')
ax.plot(t*1e6,chirp_ref.imag,label=r'$\Im$')
ax.set_ylabel('Amplitude')
ax.set_xlabel('t [$\mu$s]')
ax.set_title('chirp_ref')
ax.legend(loc='upper right')
ax.grid(True)

ax = fig.add_subplot(gs[0,1])
ax.plot(t*1e6,np.abs(chirp_ref))
ax.set_ylabel('Amplitude')
ax.set_xlabel('t [$\mu$s]')
ax.set_title('chirp_ref')
ax.grid(True)

ax = fig.add_subplot(gs[0,2])
ax.plot(t,np.degrees((np.unwrap(np.angle(chirp_ref)))))
ax.set_ylabel('deg')
ax.set_xlabel('t [$\mu$s]')
ax.set_title('Ph (chirp_ref(f))')
ax.grid(True)

#=============

ax = fig.add_subplot(gs[1,0])
ax.plot(f/1e6,chirp_ref_f.real,label=r'$\Re$')
ax.plot(f/1e6,chirp_ref_f.imag,label=r'$\Im$')
ax.set_ylabel('Amplitude')
ax.set_xlabel('Freq [MHz]')
ax.set_title('chirp_ref')
ax.legend(loc='upper right')
ax.set_xlim((-3*BW/1e6,3*BW/1e6))
ax.grid(True)

ax = fig.add_subplot(gs[1,1])
ax.plot(f/1e6,np.abs(chirp_ref_f))
ax.set_ylabel('Amplitude')
ax.set_xlabel('Freq [MHz]')
ax.set_title('|(chirp_ref(f))|')
ax.set_xlim((-3*BW/1e6,3*BW/1e6))
ax.grid(True)

ax = fig.add_subplot(gs[1,2])
ax.plot(f/1e6,np.degrees(np.unwrap((np.angle(chirp_ref_f)))))
ax.set_ylabel('Amplitude')
ax.set_xlabel('Freq [MHz]')
ax.set_title('Ph (chirp_ref(f)) ')
ax.set_xlim((-3*BW/1e6,3*BW/1e6))
ax.grid(True)

ax = fig.add_subplot(gs[2,:])
ax.specgram(chirp_ref,
            Fs=fs,
            mode='psd',
            scale='linear',
            #pad_to = 2**10,
            #NFFT = 2**8,
            )
ax.set_xlabel('time [s]')
ax.set_ylabel('Freq [Hz]')

# Plot matched filter centered in 0

fig = plt.figure(layout="tight",figsize=(16,9))
gs = GridSpec(3, 3, figure=fig)

ax = fig.add_subplot(gs[0,0])
ax.plot(t*1e6,matched_filter.real,label=r'$\Re$')
ax.plot(t*1e6,matched_filter.imag,label=r'$\Im$')
ax.set_ylabel('Amplitude')
ax.set_xlabel('Freq [MHz]')
ax.set_title('matched_filter(t)')
ax.legend(loc='upper right')
ax.grid(True)

ax = fig.add_subplot(gs[0,1])
ax.plot(t*1e6,np.abs(matched_filter))
ax.set_ylabel('Amplitude')
ax.set_xlabel('t [$\mu$s]')
ax.set_title('|matched_filter(t)|')
ax.grid(True)

ax = fig.add_subplot(gs[0,2])
ax.plot(t,np.degrees((np.unwrap(np.angle(matched_filter)))))
ax.set_ylabel('deg')
ax.set_xlabel('t [$\mu$s]')
ax.set_title('Ph (matched_filter(t))')
ax.grid(True)

#=============

ax = fig.add_subplot(gs[1,0])
ax.plot(f/1e6,matched_filter_f.real,label=r'$\Re$')
ax.plot(f/1e6,matched_filter_f.imag,label=r'$\Im$')
ax.set_ylabel('Amplitude')
ax.set_xlabel('Freq [MHz]')
ax.set_title('matched_filter(f)')
ax.legend(loc='upper right')
ax.set_xlim((-3*BW/1e6,3*BW/1e6))
ax.grid(True)

ax = fig.add_subplot(gs[1,1])
ax.plot(f/1e6,np.abs(matched_filter_f))
ax.set_ylabel('Amplitude')
ax.set_xlabel('Freq [MHz]')
ax.set_title('|(matched_filter(f))|')
ax.set_xlim((-3*BW/1e6,3*BW/1e6))
ax.grid(True)

ax = fig.add_subplot(gs[1,2])
ax.plot(f/1e6,np.degrees(np.unwrap((np.angle(matched_filter_f)))))
ax.set_ylabel('Amplitude')
ax.set_xlabel('Freq [MHz]')
ax.set_title('Ph (matched_filter(f)) ')
ax.set_xlim((-3*BW/1e6,3*BW/1e6))
ax.grid(True)

ax = fig.add_subplot(gs[2,:])
ax.specgram(matched_filter,
            Fs=fs,
            mode='psd',
            scale='linear',
            #pad_to = 2**10,
            #NFFT = 2**8,
            )
ax.set_xlabel('time [s]')
ax.set_ylabel('Freq [Hz]')

# Plot compressed signal centered in 0

fig = plt.figure(layout="tight",figsize=(16,9))
gs = GridSpec(3, 3, figure=fig)

ax = fig.add_subplot(gs[0,0])
ax.plot(t*1e6,chirp_comp.real,label=r'$\Re$')
ax.plot(t*1e6,chirp_comp.imag,label=r'$\Im$')
ax.set_ylabel('Amplitude')
ax.set_xlabel('t [$\mu$s]')
ax.set_title('chirp_comp(t)')
ax.legend(loc='upper right')
ax.grid(True)

ax = fig.add_subplot(gs[0,1])
ax.plot(t*1e6,np.abs(chirp_comp))
ax.set_ylabel('Amplitude')
ax.set_xlabel('t [$\mu$s]')
ax.set_title('|chirp_comp(t)|')
ax.grid(True)

ax = fig.add_subplot(gs[0,2])
ax.plot(t,np.degrees((np.unwrap(np.angle(chirp_comp)))))
ax.set_ylabel('deg')
ax.set_xlabel('t [$\mu$s]')
ax.set_title('Ph (chirp_comp(t))')
ax.grid(True)

#=============

ax = fig.add_subplot(gs[1,0])
ax.plot(f_full/1e6,chirp_comp_f.real,label=r'$\Re$')
ax.plot(f_full/1e6,chirp_comp_f.imag,label=r'$\Im$')
ax.set_ylabel('Amplitude')
ax.set_xlabel('Freq [MHz]')
ax.set_title('chirp_comp(f)')
ax.legend(loc='upper right')
#ax.set_xlim((-3*BW/1e6,3*BW/1e6))
ax.grid(True)

ax = fig.add_subplot(gs[1,1])
ax.plot(f_full/1e6,np.abs(chirp_comp_f))
ax.set_ylabel('Amplitude')
ax.set_xlabel('Freq [MHz]')
ax.set_title('|(chirp_comp(f))|')
#ax.set_xlim((-3*BW/1e6,3*BW/1e6))
ax.grid(True)

ax = fig.add_subplot(gs[1,2])
ax.plot(f_full/1e6,np.degrees(np.unwrap((np.angle(chirp_comp_f)))))
ax.set_ylabel('Amplitude')
ax.set_xlabel('Freq [MHz]')
ax.set_title('Ph (chirp_comp(f)) ')
#ax.set_xlim((-3*BW/1e6,3*BW/1e6))
ax.grid(True)

ax = fig.add_subplot(gs[2,:])
ax.specgram(chirp_comp,
            Fs=fs,
            mode='psd',
            scale='linear',
            #pad_to = 2**10,
            #NFFT = 2**8,
            )
ax.set_xlabel('time [s]')
ax.set_ylabel('Freq [Hz]')

#%%

# Vectors

rank_min = (Tp)*c/2 # Minimum Range
rank_max = 30e3 # Rango ficticio, deería ser el ambiguo pero se relentiza la simulación
rank_res = ts*c/2
tmax = 2*rank_max/c 

# Signals

## Simetrical Chirp Centered in Tp/2
#t = np.arange(0,tmax,ts)
#rect_tx = np.where(np.abs(t)<=Tp,1,0)

# Simetrical Chirp Centered in 0
#t = np.arange(-tmax/2,tmax/2,ts)
#rect_tx = np.where(np.abs(t)<=Tp/2,1,0)

# Asymetrical chirp (pos&neg)
#t = np.arange(-tmax/2,tmax/2,ts)
#rect_tx = np.where(((t>=(-Tp/2)*1.2) & (t<=0)) | ((t<=(Tp/2)*1.9) & (t>=0)),1,0)

# Asymetrical chirp (pos)
t = np.arange(0,tmax,ts)
rect_tx = np.where(t<(Tp/2)*1.57,1,0)



tx_chirp = np.exp(1j*np.pi*BW/Tp * t**2)
tx_chirp = rect_tx*tx_chirp
tx_chirp_f = fft(tx_chirp)
#tx_chirp_f = fftshift(fft(tx_chirp))

echo_time = np.arange(0,tmax,ts)
ranks = np.arange(rank_res,rank_max+rank_res,rank_res)
Npts = len(t)
f = fftfreq(Npts,ts)

rank = 10e3
target_rcs = 300

t_shift = 2*rank/c
rx_chirp_f = tx_chirp_f * np.exp(-1j*2*np.pi*f*t_shift) * np.exp(-1j*kwave*(2*rank))
#rx_chirp_f = fftshift(tx_chirp_f * np.exp(-1j*2*np.pi*f*t_shift) * np.exp(-1j*kwave*(2*rank)))
rx_chirp = ifft((rx_chirp_f))

#matched_filter = np.exp(-1j*np.pi*BW/Tp * t**2)
matched_filter = np.flip(np.conj(tx_chirp))
matched_filter_f = fft(matched_filter)
#matched_filter_f = matched_filter_f * np.exp(1j*2*np.pi*f*Tp)

rx_comp_f = rx_chirp_f*matched_filter_f
#rx_comp_f = fftshift(rx_chirp_f*matched_filter_f)
rx_comp = ifft(rx_comp_f)


# Powers & Noise

rect_rx = np.where(ranks>rank_min,1,0)

Gt_db = 40
Gt = 10**(Gt_db/10)
Pt = 1e6 #W
A_ant = 5*2.75
Ae = A_ant*0.6

rcs = np.abs(np.random.normal(loc=0, scale = 0.01, size=len(ranks)))
P_n_r = Pt*Gt*Ae*rcs/((4*np.pi**2)*(ranks**4))
P_n_r_rect  = P_n_r * rect_rx 

rx_comp_rect = rect_rx*(rx_comp * target_rcs * (Pt*Gt*Ae)/((4*np.pi**2)*(ranks**4)))

rx_comp_noisy = rx_comp_rect + P_n_r_rect



#fig, axes = plt.subplots(4,2)
fig = plt.figure(layout="tight",figsize=(16,9))
gs = GridSpec(5, 3, figure=fig)

ax = fig.add_subplot(gs[0,0])
ax.plot(t*1e6,np.real(tx_chirp))
ax.plot(t*1e6,np.imag(tx_chirp))
ax.plot(t*1e6,np.abs(tx_chirp))
ax.plot(t*1e6,rect_tx,'g--',lw=2)
ax.set_xlabel('Time [$\mu$s]')
ax.set_title('Tx Chirp')
#ax.set_xlim(0,150)
ax.grid(True)

ax = fig.add_subplot(gs[0,1])
ax.plot(t*1e6,np.unwrap(np.angle(tx_chirp)))
ax.set_xlabel('Time [$\mu$s]')
ax.set_title('Tx Chirp')
#ax.set_xlim(0,150)
ax.grid(True)

ax = fig.add_subplot(gs[0,2])
ax.plot(f[0:Npts//2]/1e6,np.real(tx_chirp_f[0:Npts//2]))
ax.plot(f[0:Npts//2]/1e6,np.imag(tx_chirp_f[0:Npts//2]))
ax.plot(f[0:Npts//2]/1e6,np.abs(tx_chirp_f[0:Npts//2]))
ax.set_xlabel('Freq [MHz]')
#ax.set_xlim(0,3*BW/1e6)
ax.set_title('Tx Chirp')
ax.grid(True)

ax = fig.add_subplot(gs[1,0])
ax.plot(t*1e6,np.real(rx_chirp))
ax.plot(t*1e6,np.imag(rx_chirp))
ax.plot(t*1e6,np.abs(rx_chirp))
ax.set_title('Echo Chirp')
ax.set_xlabel('Time [$\mu$s]')
#ax.set_xlim(0,150)
ax.grid(True)

ax = fig.add_subplot(gs[1,1])
ax.plot(t*1e6,np.unwrap(np.angle(rx_chirp)))
ax.set_title('Echo Chirp')
ax.set_xlabel('Time [$\mu$s]')
#ax.set_xlim(0,150)
ax.grid(True)

ax = fig.add_subplot(gs[1,2])
ax.plot(f[0:Npts//2]/1e6,np.real(rx_chirp_f[0:Npts//2]))
ax.plot(f[0:Npts//2]/1e6,np.imag(rx_chirp_f[0:Npts//2]))
ax.plot(f[0:Npts//2]/1e6,np.abs(rx_chirp_f[0:Npts//2]))
ax.set_xlabel('Freq [MHz]')
#ax.set_xlim(0,3*BW/1e6)
ax.set_title('Echo Chirp')
ax.grid(True)

ax = fig.add_subplot(gs[2,0])
ax.plot(echo_time*1e6,np.real(rx_comp))
ax.plot(echo_time*1e6,np.imag(rx_comp))
ax.plot(echo_time*1e6,np.abs(rx_comp))
ax.set_xlabel('Time [$\mu$s]')
ax.set_title('Echo Chirp Compressed')
#ax.set_xlim(0,150)
ax.grid(True)

ax = fig.add_subplot(gs[2,1])
ax.plot(echo_time*1e6,np.unwrap(np.angle(rx_comp)))
ax.set_xlabel('Time [$\mu$s]')
ax.set_title('Echo Chirp Compressed')
#ax.set_xlim(0,150)
ax.grid(True)

ax = fig.add_subplot(gs[2,2])
ax.plot(f[0:Npts//2]/1e6,np.real(rx_comp_f[0:Npts//2]))
ax.plot(f[0:Npts//2]/1e6,np.imag(rx_comp_f[0:Npts//2]))
ax.plot(f[0:Npts//2]/1e6,np.abs(rx_comp_f[0:Npts//2]))
ax.set_xlabel('Freq [MHz]')
#ax.set_xlim(0,3*BW/1e6)
ax.set_title('Echo Chirp Compressed')
ax.grid(True)

ax = fig.add_subplot(gs[3,:])
#ax.plot(ranks/1e3,np.abs(rx_comp))
ax.plot(ranks/1e3,np.abs(rx_comp_noisy))
ax.set_xlabel('Rank [km]')
ax.set_title('Echo Chirp Compressed')
#ax.set_xlim(0,150)
ax.grid(True)


ax = fig.add_subplot(gs[4,:])
ax.plot(ranks/1e3,np.abs(P_n_r_rect))
ax.set_xlabel('Rank [km]')
#ax.set_xlim(0,150)
ax.set_title('Rx Signal')
ax.grid(True)
