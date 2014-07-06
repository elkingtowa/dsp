#magnitude and phase of fourier transform

#make DFT matrix
def dftmatrix(N):
	import numpy as np

		#create 1xN matrix with indices from 0 to N-1

		a = np.expand_dims(np.arange(N), 0)

		#create matrix

		WN = np.exp(-2j*np.pi*a.T*a/N)

		return WN

%pylab inline
import numpy as np

#step function with length 128

x = np.append(np.ones(64), np.zeros(64))

y = np.linspace(0,127,128)

pylab.stem(y,x)
xlabel('Samples')
ylabel('x')
ylim([0,2])
show()

x = x.flatten();
N=128
W128 = dftmatrix(N)
X = np.dot(W128, x)

#magnitude of transformation

magnitude = abs(X)
y = np.linspace(0,127,128)
#normalize frequencies
y[:] = y[:]/len(y)
pylab.stem(y,magnitude)
ylabel('Magnitude')
xlabel('Normalized Frequencies')
show()

#phase of transformation

phase = np.angle(X)
pylab.stem(y,phase)
ylabel('Angle(X)')
xlabel('Normalized Frequencies')
show()

#theoretical phase

theoretical_phase = np.linspace(-1.5,1.5,128)
theoretical_phase[0:127:2] = 0
pylab.stem(y,theoretical_phase)
ylabel('Angle(X)')
xlabel('Normalized Frequencies')
xlim([0,1])
ylim([-2,2])
show()