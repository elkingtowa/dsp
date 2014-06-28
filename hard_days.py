# Parameters:
#
# - Fs       : sampling frequency
# - F0       : frequency of the notes forming chord
# - gain     : gains of individual notes in the chord
# - duration : duration of the chord in second
# - alpha    : attenuation in KS algorithm

Fs = 48000

import numpy as np
from karplus_strong_mat import ks
# D2, D3, F3, G3, F4, A4, C5, G5
F0 = 440*np.array([pow(2,(-31./12.)), pow(2,(-19./12.)), pow(2,(-16./12.)), pow(2,(-14./12.)), pow(2,(-4./12.)), 1, pow(2,(3./12.)), pow(2,(10./12.))])
gain = np.array([1.2, 3.0, 1.0, 2.2, 1.0, 1.0, 1.0, 3.5])
duration = 4
alpha = 0.9785

# Number of samples in the chord
nbsample_chord = Fs * duration

# This is used to correct alpha later, so that all the notes decay together
# (with the same decay rate)
first_duration = np.ceil(float(nbsample_chord)/round(float(Fs)/float(F0[0])))

# Initialization
chord = np.zeros(nbsample_chord)

for i in range(len(F0)):
    
    # Get M and duration parameter
    current_M = round(float(Fs)/float(F0[i]));
    current_duration = np.ceil(float(nbsample_chord)/float(current_M))
    
    # Correct current alpha so that all the notes decay together (with the
    # same decay rate)
    current_alpha = pow(alpha,(float(first_duration)/float(current_duration)))
    
    # Let Paul's high D on the bass ring a bit longer
    if i == 1:
        current_alpha = pow(current_alpha,8)
    
    # Generate input and output of KS algorithm
    x = np.random.rand(current_M)
    y = ks(x, current_alpha, current_duration)
    y = y[0:nbsample_chord]
    
    # Construct the chord by adding the generated note (with the
    # appropriate gain)
    chord = chord + gain[i] * y

import numpy as np
from scipy.io.wavfile import write

data = chord
scaled = np.int16(data/np.max(np.abs(data)) * 32767)

write('hard_days.wav', 44100, scaled)

import Audio
Audio.Audio(data=data, rate=48000, embed=True)