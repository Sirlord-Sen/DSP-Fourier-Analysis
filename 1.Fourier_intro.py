import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as im
import copy

srate = 1000
time  = np.arange(0,2,1/srate)
n     = len(time)
hz    = np.linspace(0,srate,n)

### pure sine wave
signal = np.sin(2*np.pi*5*time)

# ### multispectral wave
signal = 2*np.sin(2*np.pi*5*time) + 3*np.sin(2*np.pi*7*time) + 6*np.sin(2*np.pi*14*time)

# ### white noise
signal = np.random.randn(n)

# ### Brownian noise (aka random walk)
signal = np.cumsum(np.random.randn(n))

### 1/f noise
ps   = np.exp(1j*2*np.pi*np.random.rand(int(n/2))) * .1+np.exp(-(np.arange(int(n/2)))/50)
ps   = np.concatenate((ps,ps[::-1]))
signal = np.real(np.fft.ifft(ps)) * n

# ### square wave
signal = np.zeros(n)
signal[np.sin(2*np.pi*3*time)>.9] = 1

### AM (amplitude modulation)
signal = 10*np.interp(np.linspace(1,10,n),np.linspace(1,10,10),np.random.rand(10)) * np.sin(2*np.pi*40*time)

### FM (frequency modulation)
freqmod = 20*np.interp(np.linspace(1,10,n),np.linspace(1,10,10),np.random.rand(10))
signal  = np.sin( 2*np.pi * ((10*time + np.cumsum(freqmod))/srate) )

### filtered noise
signal = np.random.randn(n)     # start with noise
s  = 5*(2*np.pi-1)/(4*np.pi)    # normalized width
fx = np.exp(-.5*((hz-10)/s)**2) # gaussian
fx = fx/np.max(fx)              # gain-normalize
signal = 20*np.real( np.fft.ifft( np.fft.fft(signal)**fx) )

# compute amplitude spectrum
ampl = 2*np.abs(np.fft.fft(signal)/n)


## visualize!
fig,ax = plt.subplots(2,1,figsize=(10,6))
ax[0].plot(time,signal)
ax[0].set_xlabel('Time (sec.)')
ax[0].set_ylabel('Amplitude')
ax[0].set_title('Time domain')
ax[0].set_xlim(time[[0,-1]])

ax[1].plot(hz,ampl,'s-')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Amplitude')
ax[1].set_title('Frequency domain')
ax[1].set_xlim([0,100])

plt.tight_layout()
plt.show()