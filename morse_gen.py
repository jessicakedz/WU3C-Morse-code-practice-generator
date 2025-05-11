# simple morse code practice generator
# by WU3C 9May2025
#
import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
import random
from scipy.signal import butter, sosfilt, sosfreqz
from scipy.signal import lfilter
os.system('cls')
#============================================
# load message from this file
message_file='C:/data/CW practice/message.txt'
#=====================================
# code characteristics
wpm=20
frequency = 500  # cw freq Hz
keykClick=0.2 # 0 to 1
# Parameters
SNR=10 # signal to noise in dB
qSNR=1 #QSB depth dB
#=====================================
# channel filter
lowcut = 200.0  # Low cutoff frequency in Hz
highcut = 1200.0 # High cutoff frequency in Hz

#=======================================================================================================================================
# globals
cps=wpm*5/60
tblock = 0.4/cps  # seconds
sampling_rate = 44100
order = 4       # Filter order
noise_amplitude=0.05
#============================================================
# get text
with open(message_file, 'r') as file:
    text = file.read()
modified = text.replace('\n', '    ')  # 4 spaces
allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./ ")
message = ''.join(c for c in modified if c.upper() in allowed)

#============================================================
# Design filter (second-order sections for stability)
sos = butter(order, [lowcut, highcut], btype='band', fs=sampling_rate, output='sos')

# functions
def CWchar(mc,dit,dah,space):
    wave=np.zeros((10))
    for s in mc:
        if s==0:
            wave=np.concatenate((wave,dit))
            wave=np.concatenate((wave,space))
        if s==1:
            wave=np.concatenate((wave,dah))
            wave=np.concatenate((wave,space))
    wave=np.concatenate((wave,space))
    wave=np.concatenate((wave,space))
    wave=np.concatenate((wave,space))
    return np.array(wave)
def pink_noise(N):
    """Generate 1/f pink noise with N samples."""
    # Frequency bins
    freqs = np.fft.rfftfreq(N)
    freqs[0] = freqs[1]  # avoid division by zero

    # White noise spectrum
    spectrum = np.random.normal(size=freqs.shape) + 1j * np.random.normal(size=freqs.shape)
    
    # Apply 1/f filter
    spectrum /= np.sqrt(freqs)
    
    # Transform back to time domain
    y = np.fft.irfft(spectrum, n=N)
    y -= np.mean(y)
    y /= np.max(np.abs(y))  # normalize
    return y
def upsample(x, up_factor):
    # Step 1: Insert zeros between samples
    N = len(x)
    upsampled = np.zeros(N * up_factor)
    upsampled[::up_factor] = x

    # Step 2: Apply rectangular (moving average) filter
    w=tukey(up_factor, alpha=0.6, sym=True)
    # box = np.ones(up_factor)  # rectangular window
    y = lfilter(w, 1, upsampled)  # simple FIR filter

    return y
def impulsive_noise(N, num_spikes, spike_amplitude=1.0):
    """Generate impulsive noise of length N with random spikes."""
    up_factor=100
    x = np.zeros(int(np.floor(N/up_factor)))
    spike_positions = np.random.choice(int(np.floor(N/up_factor)), size=int(np.floor(num_spikes/up_factor)), replace=False)
    x[spike_positions] = spike_amplitude * np.random.choice([-1, 1], size=int(np.floor(num_spikes/up_factor)))
    slush=N-up_factor*int(np.floor(N/up_factor))
    slush_fill=np.zeros(slush)
    temp=0.5*np.concatenate((upsample(x, up_factor),slush_fill))
    #random amplitudes
    pn=pink_noise(np.size(temp))
    signal=np.multiply(pn,temp)
    return temp
def QSB(N,m):
    temp=pink_noise(m)
    
    slush=N-m*int(np.floor(N/m))
    slush_fill=np.zeros(slush)
    envelope=np.abs(np.concatenate((upsample(temp, int(np.floor(N/m))),slush_fill)))
    return envelope
chardict = {
    "A": np.array([0,1]),
    "B": np.array([1,0,0,0]),
    "C": np.array([1,0,1,0]),
    "D": np.array([1,0,0]),
    "E": np.array([0]),
    "F": np.array([0,0,1,0]),    
    "G": np.array([1,1,0]),
    "H": np.array([0,0,0,0]),
    "I": np.array([0,0]),
    "J": np.array([0,1,1,1]),
    "K": np.array([1,0,1]),
    "L": np.array([0,1,0,0]), 
    "M": np.array([1,1]),
    "N": np.array([1,0]),
    "O": np.array([1,1,1]),
    "P": np.array([0,1,1,0]),
    "Q": np.array([1,1,0,1]),
    "R": np.array([0,1,0]),    
    "S": np.array([0,0,0]),
    "T": np.array([1]), 
    "U": np.array([0,0,1]),
    "V": np.array([0,0,0,1]),
    "W": np.array([0,1,1]),
    "X": np.array([1,0,0,1]),
    "Y": np.array([1,0,1,1]),
    "Z": np.array([1,1,0,0]),
    "1": np.array([0,1,1,1,1]), 
    "2": np.array([0,0,1,1,1]),
    "3": np.array([0,0,0,1,1]),
    "4": np.array([0,0,0,0,1]),
    "5": np.array([0,0,0,0,0]),
    "6": np.array([1,0,0,0,0]),
    "7": np.array([1,1,0,0,0]),
    "8": np.array([1,1,1,0,0]),
    "9": np.array([1,1,1,1,0]),
    "0": np.array([1,1,1,1,1]),
    ".": np.array([0,1,0,1,0,1]),
    "/": np.array([1,0,0,1,0]),
}
# "w"
mc=np.array([0,1,1])
ditTime=0.75*tblock/3
dahTime=0.75*tblock
duration=np.size(mc)*tblock
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
# brute force first
w=tukey(int(sampling_rate * ditTime), alpha=keykClick, sym=True)

dit = np.multiply(w,0.5 * np.sin(2 * np.pi * frequency * np.linspace(0, ditTime, int(sampling_rate * ditTime), endpoint=False)))
space= 0.01* np.linspace(0, ditTime, int(sampling_rate * ditTime), endpoint=False)
w=tukey(int(sampling_rate * dahTime), alpha=keykClick, sym=True)
dah=  np.multiply(w,0.5 * np.sin(2 * np.pi * frequency * np.linspace(0, dahTime, int(sampling_rate * dahTime), endpoint=False)))
wordSpace=0.01* np.linspace(0, 7*ditTime, int(sampling_rate * 7*ditTime), endpoint=False)

wave=np.array([0,0])

for c in message.upper():
    if c == " ":
        wave=np.concatenate((wave,wordSpace))
    else:
        wave=np.concatenate((wave,CWchar(chardict[c],dit,dah,space)))

noise = noise_amplitude*np.random.normal(0,1,np.size(wave))
# make brownian noise
brown = np.cumsum(noise)
brown = brown - np.mean(brown)  # remove DC offset
brown = brown / np.max(np.abs(brown))  # normalize
#QSB
rate=sampling_rate*10
envelope=1-10**(-qSNR/20)+10**(-qSNR/20)*QSB(np.size(wave),rate)
wave=np.multiply(wave,envelope)
#add noise
nwave=np.add(noise,brown)
nwave=np.add(1*pink_noise(np.size(nwave)),nwave)
nwave=np.add(impulsive_noise(np.size(nwave), random.randint(np.floor(np.size(nwave)/10000),np.floor(np.size(nwave)/100)), spike_amplitude=2.0),nwave)
# BPF
temp=np.add(wave , 10**(-SNR/20)*nwave)
maxs=np.max(temp)
wave = sosfilt(sos, temp/maxs)
#=========================================================
sd.play(wave, sampling_rate, blocking=False)
# without blocking - ass copy test next
copy=input()
print(message)
print(copy)
# notes:
# The length of a dot is 1 time unit.
# A dash is 3 time units.
# The space between symbols (dots and dashes) of the same letter is 1 time unit.
# The space between letters is 3 time units.
# The space between words is 7 time units.